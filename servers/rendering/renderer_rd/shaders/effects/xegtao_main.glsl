///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Intel XeGTAO — Main GTAO Pass
// Consumes view-space depth mip chain and normals, outputs packed AO term + edge map.
// Adapted from Intel's XeGTAO for Godot's internal rendering pipeline.
//
// Quality modes (controlled via VERSION_DEFINES):
//   MODE_LOW    — 1 slice, 3 steps
//   MODE_MEDIUM — 2 slices, 2 steps
//   MODE_HIGH   — 3 slices, 3 steps
//   MODE_ULTRA  — 9 slices, 3 steps
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Working AO term (uint packed visibility 0..255).
layout(set = 0, binding = 0, r16ui) uniform restrict writeonly uimage2D out_working_ao;
// Packed edges (8-bit UNORM as float 0..1).
layout(set = 0, binding = 1, r8) uniform restrict writeonly image2D out_working_edges;

// View-space depth with mip chain (R16F, 5 levels).
layout(set = 1, binding = 0) uniform sampler2D src_viewspace_depth;
// Godot's normal_roughness buffer: view-space normal in [0..1] range, xyz * 2 - 1, z-flipped.
layout(rgba8, set = 1, binding = 1) uniform restrict readonly image2D source_normal;

// Push constants — 120 bytes (under 128 limit).
layout(push_constant, std430) uniform Params {
	vec2 viewport_size;           //  8
	vec2 viewport_pixel_size;     // 16

	vec2 ndc_to_view_mul;         // 24
	vec2 ndc_to_view_add;         // 32

	vec2 ndc_to_view_mul_x_pixel; // 40
	float effect_radius;          // 44
	float effect_falloff_range;   // 48

	float radius_multiplier;      // 52
	float final_value_power;      // 56
	float sample_distribution_power; // 60
	float thin_occluder_compensation; // 64

	float depth_mip_sampling_offset; // 68
	int noise_index;              // 72
	float pad0;                   // 76
	float pad1;                   // 80
}
params;

// Quality presets — slice and step counts are compile-time constants per mode.
#ifdef MODE_LOW
const int SLICE_COUNT = 1;
const int STEPS_PER_SLICE = 3;
#endif

#ifdef MODE_MEDIUM
const int SLICE_COUNT = 2;
const int STEPS_PER_SLICE = 2;
#endif

#ifdef MODE_HIGH
const int SLICE_COUNT = 3;
const int STEPS_PER_SLICE = 3;
#endif

#ifdef MODE_ULTRA
const int SLICE_COUNT = 9;
const int STEPS_PER_SLICE = 3;
#endif

const float XE_GTAO_PI = 3.1415926535897932384626433832795;
const float XE_GTAO_PI_HALF = 1.5707963267948966192313216916398;
const float XE_GTAO_OCCLUSION_TERM_SCALE = 1.5;
const int XE_GTAO_DEPTH_MIP_LEVELS = 5;

#define saturate(x) clamp((x), 0.0, 1.0)

vec3 compute_viewspace_position(vec2 screenPos01, float viewspaceDepth) {
	vec3 ret;
	ret.xy = (params.ndc_to_view_mul * screenPos01 + params.ndc_to_view_add) * viewspaceDepth;
	ret.z = viewspaceDepth;
	return ret;
}

vec4 calculate_edges(float centerZ, float leftZ, float rightZ, float topZ, float bottomZ) {
	vec4 edgesLRTB = vec4(leftZ, rightZ, topZ, bottomZ) - centerZ;

	float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
	float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
	vec4 edgesSlopeAdjusted = edgesLRTB + vec4(slopeLR, -slopeLR, slopeTB, -slopeTB);
	edgesLRTB = min(abs(edgesLRTB), abs(edgesSlopeAdjusted));

	float denom = max(centerZ * 0.011, 1e-6);
	return saturate(1.25 - edgesLRTB / denom);
}

float pack_edges(vec4 edgesLRTB) {
	edgesLRTB = round(saturate(edgesLRTB) * 2.9);
	return dot(edgesLRTB, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

// Load view-space normal from Godot's normal_roughness buffer.
// Godot stores normals as (xyz * 0.5 + 0.5) with Z pointing into the screen (negative view-space).
vec3 load_normal(ivec2 coord) {
	vec3 encoded = imageLoad(source_normal, coord).xyz;
	vec3 n = normalize(encoded * 2.0 - 1.0);
	n.z = -n.z; // Godot: -Z forward → XeGTAO: +Z forward depth
	return n;
}

// Fast sqrt/acos from Intel's implementation (bit hacks).
float fast_sqrt(float x) {
	int i = floatBitsToInt(x);
	i = int(0x1fbd1df5) + (i >> 1);
	return intBitsToFloat(i);
}

float fast_acos(float inX) {
	const float PI = 3.141593;
	const float HALF_PI = 1.570796;
	float x = abs(inX);
	float res = -0.156583 * x + HALF_PI;
	res *= fast_sqrt(max(0.0, 1.0 - x));
	return (inX >= 0.0) ? res : PI - res;
}

float fast_acos_sat(float x) {
	return fast_acos(clamp(x, -1.0, 1.0));
}

// Hilbert curve index for spatio-temporal noise (6-bit, computed inline).
uint hilbert_index(uint posX, uint posY) {
	const uint XE_HILBERT_LEVEL = 6u;
	const uint XE_HILBERT_WIDTH = (1u << XE_HILBERT_LEVEL);
	uint index = 0u;
	for (uint curLevel = XE_HILBERT_WIDTH / 2u; curLevel > 0u; curLevel /= 2u) {
		uint regionX = (posX & curLevel) > 0u ? 1u : 0u;
		uint regionY = (posY & curLevel) > 0u ? 1u : 0u;
		index += curLevel * curLevel * ((3u * regionX) ^ regionY);
		if (regionY == 0u) {
			if (regionX == 1u) {
				posX = (XE_HILBERT_WIDTH - 1u) - posX;
				posY = (XE_HILBERT_WIDTH - 1u) - posY;
			}
			uint temp = posX;
			posX = posY;
			posY = temp;
		}
	}
	return index;
}

vec2 spatio_temporal_noise(uvec2 pixCoord, uint temporalIndex) {
	uint index = hilbert_index(pixCoord.x, pixCoord.y);
	index += 288u * (temporalIndex & 63u);
	// R2 sequence
	vec2 r2 = vec2(0.75487766624669276005, 0.5698402909980532659114);
	return fract(vec2(0.5) + float(index) * r2);
}

void store_edges(ivec2 coord, float packed) {
	imageStore(out_working_edges, coord, vec4(packed, 0.0, 0.0, 0.0));
}

void store_working_ao(ivec2 coord, float visibility) {
	visibility = saturate(visibility / XE_GTAO_OCCLUSION_TERM_SCALE);
	uint packed = uint(visibility * 255.0 + 0.5);
	imageStore(out_working_ao, coord, uvec4(packed, 0u, 0u, 0u));
}

void main() {
	ivec2 pixCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 vp = ivec2(params.viewport_size);
	if (pixCoord.x >= vp.x || pixCoord.y >= vp.y) {
		return;
	}

	vec2 screenPos = (vec2(pixCoord) + 0.5) * params.viewport_pixel_size;

	// Fetch center and neighbor depths (scalar path for maximum compatibility).
	ivec2 maxPix = vp - ivec2(1);
	ivec2 pC = pixCoord;
	ivec2 pL = clamp(pixCoord + ivec2(-1, 0), ivec2(0), maxPix);
	ivec2 pR = clamp(pixCoord + ivec2(1, 0), ivec2(0), maxPix);
	ivec2 pT = clamp(pixCoord + ivec2(0, -1), ivec2(0), maxPix);
	ivec2 pB = clamp(pixCoord + ivec2(0, 1), ivec2(0), maxPix);

	float viewZ = texelFetch(src_viewspace_depth, pC, 0).r;
	float pixLZ = texelFetch(src_viewspace_depth, pL, 0).r;
	float pixRZ = texelFetch(src_viewspace_depth, pR, 0).r;
	float pixTZ = texelFetch(src_viewspace_depth, pT, 0).r;
	float pixBZ = texelFetch(src_viewspace_depth, pB, 0).r;

	// Sky / invalid depth — output fully lit, no edges.
	if (viewZ <= 1e-6) {
		store_edges(pixCoord, 0.0);
		imageStore(out_working_ao, pixCoord, uvec4(255u, 0u, 0u, 0u));
		return;
	}

	vec4 edgesLRTB = calculate_edges(viewZ, pixLZ, pixRZ, pixTZ, pixBZ);
	store_edges(pixCoord, pack_edges(edgesLRTB));

	// Load normal from Godot's normal_roughness buffer.
	vec3 viewNormal = load_normal(pC);

	// Depth format precision fix (Intel).
	viewZ *= 0.99920;

	vec3 centerPos = compute_viewspace_position(screenPos, viewZ);
	vec3 viewVec = normalize(-centerPos);

	// Dynamic AO constants.
	float effectRadius = params.effect_radius * params.radius_multiplier;
	float distributionPower = params.sample_distribution_power;
	float falloffRange = params.effect_falloff_range * effectRadius;
	float falloffFrom = effectRadius * (1.0 - params.effect_falloff_range);

	float falloffMul = -1.0 / max(falloffRange, 1e-6);
	float falloffAdd = falloffFrom / max(falloffRange, 1e-6) + 1.0;

	float visibility = 0.0;
	float effectiveSlices = 0.0;

	// Approx view-space pixel size at center depth.
	vec2 pixelDirRBViewSizeAtZ = viewZ * params.ndc_to_view_mul_x_pixel;
	float screenspaceRadius = effectRadius / max(pixelDirRBViewSizeAtZ.x, 1e-6);

	// Fade out for very small screen radii.
	visibility += saturate((10.0 - screenspaceRadius) / 100.0) * 0.5;

	const float pixelTooCloseThreshold = 1.3;

	if (screenspaceRadius <= 1e-3) {
		store_working_ao(pixCoord, 1.0);
		return;
	}

	float minS = pixelTooCloseThreshold / max(screenspaceRadius, 1e-6);

	vec2 localNoise = spatio_temporal_noise(uvec2(pixCoord), uint(params.noise_index));
	float noiseSlice = localNoise.x;
	float noiseSample = localNoise.y;

	for (int slice = 0; slice < SLICE_COUNT; slice++) {
		float sliceK = (float(slice) + noiseSlice) / float(SLICE_COUNT);
		float phi = sliceK * XE_GTAO_PI;
		float cosPhi = cos(phi);
		float sinPhi = sin(phi);
		vec2 omega = vec2(cosPhi, -sinPhi);
		omega *= screenspaceRadius;

		vec3 directionVec = vec3(cosPhi, sinPhi, 0.0);
		vec3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);

		// Robustness: skip this slice if Gram-Schmidt collapses.
		float orthoLen = length(orthoDirectionVec);
		if (orthoLen < 1e-4) {
			continue;
		}
		orthoDirectionVec /= orthoLen;

		vec3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
		vec3 projectedNormalVec = viewNormal - axisVec * dot(viewNormal, axisVec);

		float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));
		float projectedNormalLen = length(projectedNormalVec);
		float cosNorm = saturate(dot(projectedNormalVec, viewVec) / max(projectedNormalLen, 1e-6));
		float n = signNorm * fast_acos_sat(cosNorm);

		float lowHorizonCos0 = cos(n + XE_GTAO_PI_HALF);
		float lowHorizonCos1 = cos(n - XE_GTAO_PI_HALF);

		float horizonCos0 = lowHorizonCos0;
		float horizonCos1 = lowHorizonCos1;

		for (int step = 0; step < STEPS_PER_SLICE; step++) {
			float stepBaseNoise = (float(slice) + float(step) * float(STEPS_PER_SLICE)) * 0.6180339887498948482;
			float stepNoise = fract(noiseSample + stepBaseNoise);

			float s = (float(step) + stepNoise) / float(STEPS_PER_SLICE);
			s = pow(s, distributionPower);
			s += minS;

			vec2 sampleOffsetPx = s * omega;
			float sampleOffsetLen = length(sampleOffsetPx);

			float mipLevel = clamp(log2(max(sampleOffsetLen, 1e-6)) - params.depth_mip_sampling_offset, 0.0, float(XE_GTAO_DEPTH_MIP_LEVELS - 1));

			vec2 sampleOffsetUV = round(sampleOffsetPx) * params.viewport_pixel_size;

			// Side 0 (+offset)
			vec2 sampleScreenPos0 = screenPos + sampleOffsetUV;
			float SZ0 = textureLod(src_viewspace_depth, sampleScreenPos0, mipLevel).r;
			float shc0 = lowHorizonCos0;
			float weight0 = 0.0;
			if (SZ0 > 1e-6) {
				vec3 samplePos0 = compute_viewspace_position(sampleScreenPos0, SZ0);
				vec3 delta0 = samplePos0 - centerPos;
				float dist0 = max(length(delta0), 1e-6);
				vec3 hvec0 = delta0 / dist0;
				weight0 = saturate(dist0 * falloffMul + falloffAdd);
				shc0 = dot(hvec0, viewVec);
			}

			// Side 1 (-offset)
			vec2 sampleScreenPos1 = screenPos - sampleOffsetUV;
			float SZ1 = textureLod(src_viewspace_depth, sampleScreenPos1, mipLevel).r;
			float shc1 = lowHorizonCos1;
			float weight1 = 0.0;
			if (SZ1 > 1e-6) {
				vec3 samplePos1 = compute_viewspace_position(sampleScreenPos1, SZ1);
				vec3 delta1 = samplePos1 - centerPos;
				float dist1 = max(length(delta1), 1e-6);
				vec3 hvec1 = delta1 / dist1;
				weight1 = saturate(dist1 * falloffMul + falloffAdd);
				shc1 = dot(hvec1, viewVec);
			}

			// Discard unwanted samples (Intel approximation).
			shc0 = mix(lowHorizonCos0, shc0, weight0);
			shc1 = mix(lowHorizonCos1, shc1, weight1);

			horizonCos0 = max(horizonCos0, shc0);
			horizonCos1 = max(horizonCos1, shc1);
		}

		// Slope over-darkening fudge.
		projectedNormalLen = mix(projectedNormalLen, 1.0, 0.05);

		float h0 = -fast_acos_sat(horizonCos1);
		float h1 = fast_acos_sat(horizonCos0);

		float iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
		float iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;

		float localVis = projectedNormalLen * (iarc0 + iarc1);
		visibility += localVis;
		effectiveSlices += 1.0;
	}

	// Divide by effective slices (avoids darkening when slices are skipped).
	visibility /= max(effectiveSlices, 1.0);
	visibility = pow(visibility, params.final_value_power);
	visibility = max(0.0, visibility);

	store_working_ao(pixCoord, visibility);
}
