// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package anvilgpu

import _ "embed"

// CompositorHLSL is the HLSL Shader Model 5.0 source text from shaders/compositor.hlsl.
//
//go:embed shaders/compositor.hlsl
var CompositorHLSL string

// CompositorGLSL is the GLSL 330 core source text from shaders/compositor.glsl.
//
//go:embed shaders/compositor.glsl
var CompositorGLSL string

// PassesGLSL is the GLSL 330 core compositor pass suite from shaders/passes.glsl.
//
//go:embed shaders/passes.glsl
var PassesGLSL string

// PassesHLSL is the HLSL Shader Model 5.0 compositor pass suite from shaders/passes.hlsl.
//
//go:embed shaders/passes.hlsl
var PassesHLSL string

// PassEntryPoint pairs an HLSL entry point name with its Shader Model 5.0 target profile.
type PassEntryPoint struct {
	Entry   string
	Profile string
}

// PassEntryPoints lists all entry points and profiles in PassesHLSL.
var PassEntryPoints = []PassEntryPoint{
	{Entry: "VSPass", Profile: "vs_5_0"},
	{Entry: "PSShadowMask", Profile: "ps_5_0"},
	{Entry: "PSBoxBlur", Profile: "ps_5_0"},
	{Entry: "PSShadowPaint", Profile: "ps_5_0"},
	{Entry: "PSGradient", Profile: "ps_5_0"},
	{Entry: "PSBlitRgba", Profile: "ps_5_0"},
	{Entry: "PSBlitNv12", Profile: "ps_5_0"},
	{Entry: "PSBlitI420", Profile: "ps_5_0"},
}

// GLSLPasses lists all PASS_* fragment defines in PassesGLSL.
var GLSLPasses = []string{
	"PASS_SHADOW_MASK",
	"PASS_BOX_BLUR",
	"PASS_SHADOW_PAINT",
	"PASS_GRADIENT",
	"PASS_BLIT_RGBA",
	"PASS_BLIT_NV12",
	"PASS_BLIT_I420",
}
