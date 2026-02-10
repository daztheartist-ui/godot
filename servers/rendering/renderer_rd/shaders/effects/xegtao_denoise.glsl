///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Intel XeGTAO — Denoise Pass
// Edge-aware spatial denoise. Processes 2 horizontal pixels per invocation.
// Two passes: DENOISE_PASS_0 (first pass, ping) and DENOISE_PASS_1 (second pass, pong + final unpack).
// Adapted from Intel's XeGTAO for Godot's internal rendering pipeline.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef DENOISE_PASS_0
// First pass: read from working AO (from main pass), write to pong AO (R16UI).
layout(set = 0, binding = 0, r16ui) uniform restrict writeonly uimage2D out_ao_term;
layout(set = 1, binding = 0) uniform usampler2D src_ao_term;
layout(set = 1, binding = 1) uniform sampler2D src_edges;
#endif

#ifdef DENOISE_PASS_1
// Second pass: read from pong AO, write final R8 UNORM visibility.
layout(set = 0, binding = 0, r8) uniform restrict writeonly image2D out_ao_final;
layout(set = 1, binding = 0) uniform usampler2D src_ao_term;
layout(set = 1, binding = 1) uniform sampler2D src_edges;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 viewport_size;       //  8
	vec2 viewport_pixel_size;  // 16
	float denoise_blur_beta;   // 20
	uint pad0;                 // 24
	uint pad1;                 // 28
	uint pad2;                 // 32
}
params;

const float XE_GTAO_OCCLUSION_TERM_SCALE = 1.5;

#define saturate(x) clamp((x), 0.0, 1.0)

float unpack_visibility(uint packed) {
	return float(packed) / 255.0;
}

vec4 unpack_edges(float packed01) {
	uint packedVal = uint(packed01 * 255.5);
	vec4 edges;
	edges.x = float((packedVal >> 6) & 3u) / 3.0;
	edges.y = float((packedVal >> 4) & 3u) / 3.0;
	edges.z = float((packedVal >> 2) & 3u) / 3.0;
	edges.w = float((packedVal >> 0) & 3u) / 3.0;
	return saturate(edges);
}

float fetch_edge_packed(ivec2 coord) {
	ivec2 maxPix = params.viewport_size - ivec2(1);
	ivec2 c = clamp(coord, ivec2(0), maxPix);
	return texelFetch(src_edges, c, 0).r;
}

float fetch_ao(ivec2 coord) {
	ivec2 maxPix = params.viewport_size - ivec2(1);
	ivec2 c = clamp(coord, ivec2(0), maxPix);
	uint packed = texelFetch(src_ao_term, c, 0).r;
	return unpack_visibility(packed);
}

void add_sample(float value, float weight, inout float sum, inout float sumWeight) {
	sum += weight * value;
	sumWeight += weight;
}

void main() {
	ivec2 dispatchID = ivec2(gl_GlobalInvocationID.xy);
	ivec2 pixCoordBase = dispatchID * ivec2(2, 1);

	ivec2 vp = params.viewport_size;
	if (pixCoordBase.y >= vp.y || pixCoordBase.x >= vp.x) {
		return;
	}

#ifdef DENOISE_PASS_1
	bool finalApply = true;
#else
	bool finalApply = false;
#endif

	float beta = params.denoise_blur_beta;
	float blurAmount = finalApply ? beta : (beta / 5.0);
	const float diagWeight = 0.85 * 0.5;

	for (int side = 0; side < 2; side++) {
		ivec2 p = pixCoordBase + ivec2(side, 0);
		if (p.x >= vp.x) {
			continue;
		}

		// Scalar fetch path (maximum compatibility).
		vec4 edgesC = unpack_edges(fetch_edge_packed(p));
		vec4 edgesL = unpack_edges(fetch_edge_packed(p + ivec2(-1, 0)));
		vec4 edgesR = unpack_edges(fetch_edge_packed(p + ivec2(1, 0)));
		vec4 edgesT = unpack_edges(fetch_edge_packed(p + ivec2(0, -1)));
		vec4 edgesB = unpack_edges(fetch_edge_packed(p + ivec2(0, 1)));

		float vC = fetch_ao(p);
		float vL = fetch_ao(p + ivec2(-1, 0));
		float vR = fetch_ao(p + ivec2(1, 0));
		float vT = fetch_ao(p + ivec2(0, -1));
		float vBv = fetch_ao(p + ivec2(0, 1));
		float vTL = fetch_ao(p + ivec2(-1, -1));
		float vTR = fetch_ao(p + ivec2(1, -1));
		float vBL = fetch_ao(p + ivec2(-1, 1));
		float vBR = fetch_ao(p + ivec2(1, 1));

		// Enforce symmetry (Intel).
		edgesC *= vec4(edgesL.y, edgesR.x, edgesT.w, edgesB.z);

		// Allow a small amount of leaking to reduce aliasing (Intel).
		const float leak_threshold = 2.5;
		const float leak_strength = 0.5;
		float edginess = (saturate(4.0 - leak_threshold - dot(edgesC, vec4(1.0))) / (4.0 - leak_threshold)) * leak_strength;
		edgesC = saturate(edgesC + edginess);

		// Diagonal weights.
		float weightTL = diagWeight * (edgesC.x * edgesL.z + edgesC.z * edgesT.x);
		float weightTR = diagWeight * (edgesC.z * edgesT.y + edgesC.y * edgesR.z);
		float weightBL = diagWeight * (edgesC.w * edgesB.x + edgesC.x * edgesL.w);
		float weightBR = diagWeight * (edgesC.y * edgesR.w + edgesC.w * edgesB.y);

		float sumWeight = blurAmount;
		float sum = vC * sumWeight;

		add_sample(vL, edgesC.x, sum, sumWeight);
		add_sample(vR, edgesC.y, sum, sumWeight);
		add_sample(vT, edgesC.z, sum, sumWeight);
		add_sample(vBv, edgesC.w, sum, sumWeight);

		add_sample(vTL, weightTL, sum, sumWeight);
		add_sample(vTR, weightTR, sum, sumWeight);
		add_sample(vBL, weightBL, sum, sumWeight);
		add_sample(vBR, weightBR, sum, sumWeight);

		float outVal = sum / max(sumWeight, 1e-6);

#ifdef DENOISE_PASS_1
		// Final pass: apply occlusion term scale and write as R8 UNORM (Godot's expected format).
		float finalVis = saturate(outVal * XE_GTAO_OCCLUSION_TERM_SCALE);
		imageStore(out_ao_final, p, vec4(finalVis, 0.0, 0.0, 0.0));
#else
		// First pass: store back as uint-packed R16UI for ping-pong.
		uint packed = uint(outVal * 255.0 + 0.5);
		imageStore(out_ao_term, p, uvec4(packed, 0u, 0u, 0u));
#endif
	}
}
