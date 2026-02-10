///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Intel XeGTAO — Depth Prefilter Pass
// Generates a 5-level view-space depth mip chain from Godot's raw depth buffer.
// Adapted from Intel's XeGTAO for Godot's internal rendering pipeline.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output depth mips (view-space depth, fp16).
layout(set = 0, binding = 0, r16f) uniform restrict writeonly image2D out_depth_mip0;
layout(set = 0, binding = 1, r16f) uniform restrict writeonly image2D out_depth_mip1;
layout(set = 0, binding = 2, r16f) uniform restrict writeonly image2D out_depth_mip2;
layout(set = 0, binding = 3, r16f) uniform restrict writeonly image2D out_depth_mip3;
layout(set = 0, binding = 4, r16f) uniform restrict writeonly image2D out_depth_mip4;

// Godot's depth buffer (reversed-Z: near=1, far=0).
layout(set = 1, binding = 0) uniform sampler2D src_raw_depth;

layout(push_constant, std430) uniform Params {
	ivec2 viewport_size;
	vec2 viewport_pixel_size;

	// XeGTAO effect parameters needed for depth mip filter.
	float effect_radius;
	float effect_falloff_range;
	float radius_multiplier;
	float pad0;

	// Inverse projection matrix for depth linearization.
	// Stored as 4 vec4s = 64 bytes; total push constant = 96 bytes (under 128 limit).
	mat4 inv_proj;
}
params;

const int XE_GTAO_DEPTH_MIP_LEVELS = 5;
const float FP16_MAX = 65504.0;

shared float g_scratchDepths[8][8];

float clamp_depth(float depth) {
	return clamp(depth, 0.0, FP16_MAX);
}

// Godot uses reverse-Z depth (near=1, far=0). Treat cleared depth (exact 0) as invalid.
bool depth_valid(float screenDepth) {
	return screenDepth > 0.000001;
}

// Reconstruct view-space Z using the inverse projection matrix.
// View-space Z is negative in front of camera in Godot; we return positive (distance).
float screen_to_view_depth(ivec2 pixCoord, float screenDepth) {
	vec2 uv = (vec2(pixCoord) + 0.5) * params.viewport_pixel_size;
	vec4 clip = vec4(uv * 2.0 - 1.0, screenDepth, 1.0);
	vec4 view = params.inv_proj * clip;
	float iw = (abs(view.w) > 1e-6) ? (1.0 / view.w) : 0.0;
	float viewZ = -(view.z * iw);
	return clamp_depth(viewZ);
}

// Weighted depth mip filter from Intel XeGTAO (preserves nearby occluders).
float depth_mip_filter(float depth0, float depth1, float depth2, float depth3) {
	float maxDepth = max(max(depth0, depth1), max(depth2, depth3));

	const float depthRangeScaleFactor = 0.75;

	float effectRadius = depthRangeScaleFactor * params.effect_radius * params.radius_multiplier;
	float falloffRange = params.effect_falloff_range * effectRadius;
	float falloffFrom = effectRadius * (1.0 - params.effect_falloff_range);

	float falloffMul = -1.0 / max(falloffRange, 1e-6);
	float falloffAdd = falloffFrom / max(falloffRange, 1e-6) + 1.0;

	float w0 = clamp((maxDepth - depth0) * falloffMul + falloffAdd, 0.0, 1.0);
	float w1 = clamp((maxDepth - depth1) * falloffMul + falloffAdd, 0.0, 1.0);
	float w2 = clamp((maxDepth - depth2) * falloffMul + falloffAdd, 0.0, 1.0);
	float w3 = clamp((maxDepth - depth3) * falloffMul + falloffAdd, 0.0, 1.0);

	float wsum = w0 + w1 + w2 + w3;
	return (w0 * depth0 + w1 * depth1 + w2 * depth2 + w3 * depth3) / max(wsum, 1e-6);
}

#define STORE_R16F(_img, _coord, _v) imageStore((_img), (_coord), vec4((_v), 0.0, 0.0, 0.0))

void main() {
	ivec2 dispatchThreadID = ivec2(gl_GlobalInvocationID.xy);
	ivec2 groupThreadID = ivec2(gl_LocalInvocationID.xy);

	ivec2 vp = params.viewport_size;
	if (vp.x <= 0 || vp.y <= 0) {
		return;
	}

	ivec2 mip1Size = (vp + ivec2(1)) / 2;
	if (dispatchThreadID.x >= mip1Size.x || dispatchThreadID.y >= mip1Size.y) {
		return;
	}

	// MIP 0: each thread writes a 2x2 block of full-res view-space depth.
	ivec2 baseCoord = dispatchThreadID;
	ivec2 pixCoord = baseCoord * 2;

	ivec2 maxPix = vp - ivec2(1);
	ivec2 p00 = clamp(pixCoord + ivec2(0, 0), ivec2(0), maxPix);
	ivec2 p10 = clamp(pixCoord + ivec2(1, 0), ivec2(0), maxPix);
	ivec2 p01 = clamp(pixCoord + ivec2(0, 1), ivec2(0), maxPix);
	ivec2 p11 = clamp(pixCoord + ivec2(1, 1), ivec2(0), maxPix);

	float sd00 = texelFetch(src_raw_depth, p00, 0).r;
	float sd10 = texelFetch(src_raw_depth, p10, 0).r;
	float sd01 = texelFetch(src_raw_depth, p01, 0).r;
	float sd11 = texelFetch(src_raw_depth, p11, 0).r;

	// Store 0 for invalid depth so later passes skip AO on sky/cleared pixels.
	float depth0 = depth_valid(sd00) ? screen_to_view_depth(p00, sd00) : 0.0;
	float depth1 = depth_valid(sd10) ? screen_to_view_depth(p10, sd10) : 0.0;
	float depth2 = depth_valid(sd01) ? screen_to_view_depth(p01, sd01) : 0.0;
	float depth3 = depth_valid(sd11) ? screen_to_view_depth(p11, sd11) : 0.0;

	if (pixCoord.x + 0 < vp.x && pixCoord.y + 0 < vp.y) STORE_R16F(out_depth_mip0, pixCoord + ivec2(0, 0), depth0);
	if (pixCoord.x + 1 < vp.x && pixCoord.y + 0 < vp.y) STORE_R16F(out_depth_mip0, pixCoord + ivec2(1, 0), depth1);
	if (pixCoord.x + 0 < vp.x && pixCoord.y + 1 < vp.y) STORE_R16F(out_depth_mip0, pixCoord + ivec2(0, 1), depth2);
	if (pixCoord.x + 1 < vp.x && pixCoord.y + 1 < vp.y) STORE_R16F(out_depth_mip0, pixCoord + ivec2(1, 1), depth3);

	// MIP 1
	float dm1 = depth_mip_filter(depth0, depth1, depth2, depth3);
	STORE_R16F(out_depth_mip1, baseCoord, dm1);
	g_scratchDepths[groupThreadID.x][groupThreadID.y] = dm1;

	barrier();

	// MIP 2
	ivec2 mip2Size = (mip1Size + ivec2(1)) / 2;
	if (((groupThreadID.x & 1) == 0) && ((groupThreadID.y & 1) == 0)) {
		float inTL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 0];
		float inTR = g_scratchDepths[groupThreadID.x + 1][groupThreadID.y + 0];
		float inBL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 1];
		float inBR = g_scratchDepths[groupThreadID.x + 1][groupThreadID.y + 1];
		float dm2 = depth_mip_filter(inTL, inTR, inBL, inBR);
		ivec2 dst = baseCoord / 2;
		if (dst.x < mip2Size.x && dst.y < mip2Size.y) {
			STORE_R16F(out_depth_mip2, dst, dm2);
		}
		g_scratchDepths[groupThreadID.x][groupThreadID.y] = dm2;
	}

	barrier();

	// MIP 3
	ivec2 mip3Size = (mip2Size + ivec2(1)) / 2;
	if (((groupThreadID.x & 3) == 0) && ((groupThreadID.y & 3) == 0)) {
		float inTL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 0];
		float inTR = g_scratchDepths[groupThreadID.x + 2][groupThreadID.y + 0];
		float inBL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 2];
		float inBR = g_scratchDepths[groupThreadID.x + 2][groupThreadID.y + 2];
		float dm3 = depth_mip_filter(inTL, inTR, inBL, inBR);
		ivec2 dst = baseCoord / 4;
		if (dst.x < mip3Size.x && dst.y < mip3Size.y) {
			STORE_R16F(out_depth_mip3, dst, dm3);
		}
		g_scratchDepths[groupThreadID.x][groupThreadID.y] = dm3;
	}

	barrier();

	// MIP 4
	ivec2 mip4Size = (mip3Size + ivec2(1)) / 2;
	if (((groupThreadID.x & 7) == 0) && ((groupThreadID.y & 7) == 0)) {
		float inTL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 0];
		float inTR = g_scratchDepths[groupThreadID.x + 4][groupThreadID.y + 0];
		float inBL = g_scratchDepths[groupThreadID.x + 0][groupThreadID.y + 4];
		float inBR = g_scratchDepths[groupThreadID.x + 4][groupThreadID.y + 4];
		float dm4 = depth_mip_filter(inTL, inTR, inBL, inBR);
		ivec2 dst = baseCoord / 8;
		if (dst.x < mip4Size.x && dst.y < mip4Size.y) {
			STORE_R16F(out_depth_mip4, dst, dm4);
		}
	}
}
