// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilgpu

import "math"

const (
	// MaxSide is the largest accepted target, frame, or blit side in pixels.
	MaxSide uint32 = 16384
	// MaxBlurRadius is the maximum accepted CSS box-shadow blur radius in pixels.
	MaxBlurRadius float32 = 256.0
	// MaxGradientStops is the maximum number of stops accepted in a GradientDesc.
	MaxGradientStops = 16

	maskOne     uint32 = 1 << 15
	bilinearOne uint64 = 4096
)

// ComposeErrorCode identifies why a compositor draw was refused.
type ComposeErrorCode uint8

const (
	// ErrCodeBadTarget indicates target size, stride, or buffer length is unusable.
	ErrCodeBadTarget ComposeErrorCode = 1
	// ErrCodeBadFrame indicates video frame size, stride, or plane length is unusable.
	ErrCodeBadFrame ComposeErrorCode = 2
	// ErrCodeBadInput indicates a coordinate, radius, size, or stop is non-finite or out of range.
	ErrCodeBadInput ComposeErrorCode = 3
	// ErrCodeDevice indicates a hardware backend failed (Compositor automatically falls back to SoftwareBackend).
	ErrCodeDevice ComposeErrorCode = 4
)

// ComposeError is returned when a compositor operation receives invalid target, frame, or draw parameters.
type ComposeError struct {
	Code ComposeErrorCode
}

func (e *ComposeError) Error() string {
	switch e.Code {
	case ErrCodeBadTarget:
		return "target size, stride, or length is unusable"
	case ErrCodeBadFrame:
		return "frame size, stride, or plane length is unusable"
	case ErrCodeBadInput:
		return "draw input is not finite or out of range"
	case ErrCodeDevice:
		return "device backend failed"
	default:
		return "unknown compose error"
	}
}

var (
	ErrBadTarget = &ComposeError{Code: ErrCodeBadTarget}
	ErrBadFrame  = &ComposeError{Code: ErrCodeBadFrame}
	ErrBadInput  = &ComposeError{Code: ErrCodeBadInput}
	ErrDevice    = &ComposeError{Code: ErrCodeDevice}
)

// RgbaTarget wraps a straight RGBA8 destination buffer with a row stride in bytes.
type RgbaTarget struct {
	pixels []byte
	width  uint32
	height uint32
	stride int
}

// NewRgbaTarget wraps pixels as width by height with stride bytes per row.
func NewRgbaTarget(pixels []byte, width, height uint32, stride int) (*RgbaTarget, error) {
	if width > MaxSide || height > MaxSide {
		return nil, ErrBadTarget
	}
	row := int(width) * 4
	if stride < row {
		return nil, ErrBadTarget
	}
	if width > 0 && height > 0 {
		needed := stride*int(height-1) + row
		if needed < row || len(pixels) < needed {
			return nil, ErrBadTarget
		}
	}
	return &RgbaTarget{
		pixels: pixels,
		width:  width,
		height: height,
		stride: stride,
	}, nil
}

// NewPackedRgbaTarget wraps tightly packed RGBA8 pixels (stride == width * 4).
func NewPackedRgbaTarget(pixels []byte, width, height uint32) (*RgbaTarget, error) {
	return NewRgbaTarget(pixels, width, height, int(width)*4)
}

func (t *RgbaTarget) Width() uint32  { return t.width }
func (t *RgbaTarget) Height() uint32 { return t.height }
func (t *RgbaTarget) Stride() int    { return t.stride }
func (t *RgbaTarget) Pixels() []byte { return t.pixels }

// Pixel returns the four RGBA8 bytes at (x, y), or false if out of bounds.
func (t *RgbaTarget) Pixel(x, y uint32) ([4]byte, bool) {
	if x >= t.width || y >= t.height {
		return [4]byte{}, false
	}
	i := t.offset(x, y)
	return [4]byte{t.pixels[i], t.pixels[i+1], t.pixels[i+2], t.pixels[i+3]}, true
}

func (t *RgbaTarget) offset(x, y uint32) int {
	return int(y)*t.stride + int(x)*4
}

func (t *RgbaTarget) pxMut(x, y uint32) []byte {
	i := t.offset(x, y)
	return t.pixels[i : i+4]
}

func (t *RgbaTarget) save(b Bounds) []byte {
	row := int(b.Width()) * 4
	out := make([]byte, 0, row*int(b.Height()))
	for y := b.Y0; y < b.Y1; y++ {
		i := t.offset(b.X0, y)
		out = append(out, t.pixels[i:i+row]...)
	}
	return out
}

func (t *RgbaTarget) restore(b Bounds, saved []byte) {
	row := int(b.Width()) * 4
	for n, y := 0, b.Y0; y < b.Y1; n, y = n+1, y+1 {
		i := t.offset(b.X0, y)
		copy(t.pixels[i:i+row], saved[n*row:(n+1)*row])
	}
}

// Bounds is a pixel rectangle inside a target where X1 and Y1 are exclusive.
type Bounds struct {
	X0, Y0, X1, Y1 uint32
}

func (b Bounds) Width() uint32  { return b.X1 - b.X0 }
func (b Bounds) Height() uint32 { return b.Y1 - b.Y0 }

func (b Bounds) intersect(other Bounds) (Bounds, bool) {
	res := Bounds{
		X0: max(b.X0, other.X0),
		Y0: max(b.Y0, other.Y0),
		X1: min(b.X1, other.X1),
		Y1: min(b.Y1, other.Y1),
	}
	if res.X0 < res.X1 && res.Y0 < res.Y1 {
		return res, true
	}
	return Bounds{}, false
}

func isFiniteF32(v float32) bool {
	f := float64(v)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

func centerSpan(lo, hi float64, limit uint32) (uint32, uint32) {
	clamp := func(v float64) uint32 {
		if v < 0.0 {
			return 0
		}
		if v > float64(limit) {
			return limit
		}
		return uint32(v)
	}
	return clamp(math.Ceil(lo - 0.5)), clamp(math.Ceil(hi - 0.5))
}

func rectBounds(rect GpuRect, width, height uint32) (Bounds, bool) {
	x0, x1 := centerSpan(float64(rect.X), float64(rect.X)+float64(rect.Width), width)
	y0, y1 := centerSpan(float64(rect.Y), float64(rect.Y)+float64(rect.Height), height)
	if x0 < x1 && y0 < y1 {
		return Bounds{X0: x0, Y0: y0, X1: x1, Y1: y1}, true
	}
	return Bounds{}, false
}

func rectIsValid(rect GpuRect) bool {
	return isFiniteF32(rect.X) &&
		isFiniteF32(rect.Y) &&
		isFiniteF32(rect.Width) &&
		isFiniteF32(rect.Height) &&
		rect.Width >= 0.0 &&
		rect.Height >= 0.0
}

func targetBounds(target *RgbaTarget, clip *GpuRect) (Bounds, bool, error) {
	all := Bounds{X0: 0, Y0: 0, X1: target.width, Y1: target.height}
	if all.X1 == 0 || all.Y1 == 0 {
		return Bounds{}, false, nil
	}
	if clip == nil {
		return all, true, nil
	}
	if !rectIsValid(*clip) {
		return Bounds{}, false, ErrBadInput
	}
	b, ok := rectBounds(*clip, target.width, target.height)
	return b, ok, nil
}

// OverStraight composites src with coverage alpha (0..1) onto dst using straight-alpha source-over.
func OverStraight(dst []byte, src GpuColor, alpha float32) {
	sa := (float32(src.A) / 255.0) * clampF32(alpha, 0.0, 1.0)
	if sa <= 0.0 {
		return
	}
	da := float32(dst[3]) / 255.0
	keep := da * (1.0 - sa)
	oa := sa + keep
	mix := func(s, d uint8) uint8 {
		v := floorF32((float32(s)*sa+float32(d)*keep)/oa + 0.5)
		return uint8(clampF32(v, 0.0, 255.0))
	}
	dst[0] = mix(src.R, dst[0])
	dst[1] = mix(src.G, dst[1])
	dst[2] = mix(src.B, dst[2])
	dst[3] = uint8(clampF32(floorF32(oa*255.0+0.5), 0.0, 255.0))
}

// ShadowDesc describes a CSS-style box shadow.
type ShadowDesc struct {
	Rect         GpuRect
	CornerRadius float32
	OffsetX      float32
	OffsetY      float32
	BlurRadius   float32
	Spread       float32
	Color        GpuColor
	Clip         *GpuRect
}

// NewShadowDesc constructs a sharp shadow directly under rect.
func NewShadowDesc(rect GpuRect, color GpuColor) ShadowDesc {
	return ShadowDesc{
		Rect:  rect,
		Color: color,
	}
}

func (d *ShadowDesc) isValid() bool {
	return rectIsValid(d.Rect) &&
		isFiniteF32(d.CornerRadius) &&
		isFiniteF32(d.OffsetX) &&
		isFiniteF32(d.OffsetY) &&
		isFiniteF32(d.BlurRadius) &&
		isFiniteF32(d.Spread) &&
		d.CornerRadius >= 0.0 &&
		d.BlurRadius >= 0.0
}

// ShadowPassRadius returns the box blur radius for each of the three box blur passes.
func ShadowPassRadius(blurRadius float32) uint32 {
	blur := clampF32(blurRadius, 0.0, MaxBlurRadius)
	w := float32(math.Sqrt(float64(blur*blur + 1.0)))
	return uint32(roundHalfAwayFromZeroF32((w - 1.0) * 0.5))
}

type shadowPlan struct {
	out    Bounds
	maskX  int64
	maskY  int64
	maskW  int
	maskH  int
	radius int
	cx     float32
	cy     float32
	halfW  float32
	halfH  float32
	corner float32
}

func planShadow(target *RgbaTarget, desc *ShadowDesc) (*shadowPlan, error) {
	if !desc.isValid() {
		return nil, ErrBadInput
	}
	area, ok, err := targetBounds(target, desc.Clip)
	if err != nil || !ok {
		return nil, err
	}
	if desc.Color.A == 0 {
		return nil, nil
	}
	x := float64(desc.Rect.X) + float64(desc.OffsetX) - float64(desc.Spread)
	y := float64(desc.Rect.Y) + float64(desc.OffsetY) - float64(desc.Spread)
	w := float64(desc.Rect.Width) + 2.0*float64(desc.Spread)
	h := float64(desc.Rect.Height) + 2.0*float64(desc.Spread)
	if w <= 0.0 || h <= 0.0 {
		return nil, nil
	}
	radius := int(ShadowPassRadius(desc.BlurRadius))
	reach := 1.0 + 3.0*float64(radius)
	x0, x1 := centerSpan(x-reach, x+w+reach, target.width)
	y0, y1 := centerSpan(y-reach, y+h+reach, target.height)
	if x0 >= x1 || y0 >= y1 {
		return nil, nil
	}
	out, ok := (Bounds{X0: x0, Y0: y0, X1: x1, Y1: y1}).intersect(area)
	if !ok {
		return nil, nil
	}
	margin := 3 * radius
	var corner float32
	if desc.CornerRadius > 0.0 {
		corner = maxF32(desc.CornerRadius+desc.Spread, 0.0)
	}
	return &shadowPlan{
		out:    out,
		maskX:  int64(out.X0) - int64(margin),
		maskY:  int64(out.Y0) - int64(margin),
		maskW:  int(out.Width()) + 2*margin,
		maskH:  int(out.Height()) + 2*margin,
		radius: radius,
		cx:     float32(x + w*0.5),
		cy:     float32(y + h*0.5),
		halfW:  float32(w * 0.5),
		halfH:  float32(h * 0.5),
		corner: corner,
	}, nil
}

func maskValue(coverage float32) uint32 {
	return uint32(math.RoundToEven(float64(coverage * float32(maskOne))))
}

type boxRun struct {
	n       int
	r       int
	width   uint32
	acc     uint32
	ring    []uint32
	pushed  int
	emitted int
}

func newBoxRun(n, r int) boxRun {
	return boxRun{
		n:     n,
		r:     r,
		width: uint32(2*r + 1),
		ring:  make([]uint32, 2*r+2),
	}
}

func (b *boxRun) input(k int) uint32 {
	return b.ring[k%len(b.ring)]
}

func (b *boxRun) emit() uint32 {
	b.emitted++
	return (2*b.acc + b.width) / (2 * b.width)
}

func (b *boxRun) push(v uint32) (uint32, bool) {
	k := b.pushed
	slot := k % len(b.ring)
	b.pushed++
	first := min(b.r, b.n-1)
	if k <= first {
		b.acc += v
		b.ring[slot] = v
		if k == first {
			return b.emit(), true
		}
		return 0, false
	}
	i := k - b.r - 1
	b.acc += v
	if i >= b.r {
		b.acc -= b.input(i - b.r)
	}
	b.ring[slot] = v
	return b.emit(), true
}

func (b *boxRun) finish() (uint32, bool) {
	if b.emitted >= b.n {
		return 0, false
	}
	i := b.emitted - 1
	if i >= b.r {
		b.acc -= b.input(i - b.r)
	}
	return b.emit(), true
}

func boxBlurLine(src, dst []uint32, r int) {
	if len(src) == 0 {
		return
	}
	run := newBoxRun(len(src), r)
	out := 0
	for _, v := range src {
		if o, ok := run.push(v); ok {
			dst[out] = o
			out++
		}
	}
	for {
		o, ok := run.finish()
		if !ok {
			break
		}
		dst[out] = o
		out++
	}
}

func throughBoxPasses(passes []boxRun, v uint32) (uint32, bool) {
	for i := range passes {
		var ok bool
		v, ok = passes[i].push(v)
		if !ok {
			return 0, false
		}
	}
	return v, true
}

func drainBoxPasses(passes []boxRun) (uint32, bool) {
	for k := 0; k < len(passes); k++ {
		for {
			v, ok := passes[k].finish()
			if !ok {
				break
			}
			if o, ok := throughBoxPasses(passes[k+1:], v); ok {
				return o, true
			}
		}
	}
	return 0, false
}

// DrawBoxShadow draws a blurred or sharp CSS box shadow in software.
func DrawBoxShadow(target *RgbaTarget, desc *ShadowDesc) error {
	plan, err := planShadow(target, desc)
	if err != nil || plan == nil {
		return err
	}
	margin := 3 * plan.radius
	outW := int(plan.out.Width())
	coverage := make([]float32, plan.maskW)
	a := make([]uint32, plan.maskW)
	b := make([]uint32, plan.maskW)

	computeCoverage := func(j int) {
		sy := float32(plan.maskY+int64(j)) + 0.5 - plan.cy
		for i := range coverage {
			sx := float32(plan.maskX+int64(i)) + 0.5 - plan.cx
			d := SDFRoundedBox(sx, sy, plan.halfW, plan.halfH, plan.corner)
			coverage[i] = clampF32(0.5-d, 0.0, 1.0)
		}
	}

	paintRow := func(j int, line []float32) {
		if j < margin || j >= margin+int(plan.out.Height()) {
			return
		}
		y := plan.out.Y0 + uint32(j-margin)
		for n, cov := range line {
			if cov > 0.0 {
				OverStraight(target.pxMut(plan.out.X0+uint32(n), y), desc.Color, cov)
			}
		}
	}

	if plan.radius == 0 {
		for j := 0; j < plan.maskH; j++ {
			computeCoverage(j)
			paintRow(j, coverage[margin:margin+outW])
		}
		return nil
	}

	columns := make([][3]boxRun, outW)
	for n := 0; n < outW; n++ {
		columns[n] = [3]boxRun{
			newBoxRun(plan.maskH, plan.radius),
			newBoxRun(plan.maskH, plan.radius),
			newBoxRun(plan.maskH, plan.radius),
		}
	}
	line := make([]float32, outW)
	toCoverage := func(q uint32) float32 {
		return float32(q) / float32(maskOne)
	}
	next := 0
	for j := 0; j < plan.maskH; j++ {
		computeCoverage(j)
		for i, c := range coverage {
			a[i] = maskValue(c)
		}
		boxBlurLine(a, b, plan.radius)
		boxBlurLine(b, a, plan.radius)
		boxBlurLine(a, b, plan.radius)

		ready := false
		for n := 0; n < outW; n++ {
			if v, ok := throughBoxPasses(columns[n][:], b[margin+n]); ok {
				line[n] = toCoverage(v)
				ready = true
			}
		}
		if ready {
			paintRow(next, line)
			next++
		}
	}
	for next < plan.maskH {
		for n := 0; n < outW; n++ {
			v, _ := drainBoxPasses(columns[n][:])
			line[n] = toCoverage(v)
		}
		paintRow(next, line)
		next++
	}
	return nil
}

// GradientShapeKind identifies whether a GradientShape is linear or radial.
type GradientShapeKind uint8

const (
	GradientShapeLinear GradientShapeKind = iota
	GradientShapeRadial
)

// GradientShape defines the geometry of a linear or radial gradient.
type GradientShape struct {
	Kind             GradientShapeKind
	StartX, StartY   float32
	EndX, EndY       float32
	CenterX, CenterY float32
	RadiusX, RadiusY float32
}

// LinearGradientShape constructs a linear GradientShape from (startX, startY) to (endX, endY).
func LinearGradientShape(startX, startY, endX, endY float32) GradientShape {
	return GradientShape{
		Kind:   GradientShapeLinear,
		StartX: startX,
		StartY: startY,
		EndX:   endX,
		EndY:   endY,
	}
}

// RadialGradientShape constructs a radial GradientShape centered at (centerX, centerY) with radii (radiusX, radiusY).
func RadialGradientShape(centerX, centerY, radiusX, radiusY float32) GradientShape {
	return GradientShape{
		Kind:    GradientShapeRadial,
		CenterX: centerX,
		CenterY: centerY,
		RadiusX: radiusX,
		RadiusY: radiusY,
	}
}

// GradientDesc describes a linear or radial gradient filling Rect.
type GradientDesc struct {
	Rect  GpuRect
	Shape GradientShape
	Stops []GpuGradientStop
	Clip  *GpuRect
}

func validateStops(stops []GpuGradientStop) ([]GpuGradientStop, bool) {
	if len(stops) == 0 || len(stops) > MaxGradientStops {
		return nil, false
	}
	out := make([]GpuGradientStop, len(stops))
	var floor float32
	for i, s := range stops {
		if !isFiniteF32(s.Offset) {
			return nil, false
		}
		floor = maxF32(clampF32(s.Offset, 0.0, 1.0), floor)
		out[i] = NewGpuGradientStop(floor, s.Color)
	}
	return out, true
}

func planGradient(target *RgbaTarget, desc *GradientDesc) (Bounds, []GpuGradientStop, bool, error) {
	if !rectIsValid(desc.Rect) {
		return Bounds{}, nil, false, ErrBadInput
	}
	var shapeOK bool
	if desc.Shape.Kind == GradientShapeLinear {
		shapeOK = isFiniteF32(desc.Shape.StartX) && isFiniteF32(desc.Shape.StartY) &&
			isFiniteF32(desc.Shape.EndX) && isFiniteF32(desc.Shape.EndY)
	} else {
		shapeOK = isFiniteF32(desc.Shape.CenterX) && isFiniteF32(desc.Shape.CenterY) &&
			isFiniteF32(desc.Shape.RadiusX) && isFiniteF32(desc.Shape.RadiusY) &&
			desc.Shape.RadiusX >= 0.0 && desc.Shape.RadiusY >= 0.0
	}
	if !shapeOK {
		return Bounds{}, nil, false, ErrBadInput
	}
	stops, ok := validateStops(desc.Stops)
	if !ok {
		return Bounds{}, nil, false, ErrBadInput
	}
	area, ok, err := targetBounds(target, desc.Clip)
	if err != nil || !ok {
		return Bounds{}, nil, false, err
	}
	fill, ok := rectBounds(desc.Rect, target.width, target.height)
	if !ok {
		return Bounds{}, nil, false, nil
	}
	b, ok := fill.intersect(area)
	if !ok {
		return Bounds{}, nil, false, nil
	}
	return b, stops, true, nil
}

// LinearGradientT computes the stop parameter t (0..1) of a linear gradient at (px, py).
func LinearGradientT(startX, startY, endX, endY, px, py float32) float32 {
	dx := endX - startX
	dy := endY - startY
	len2 := dx*dx + dy*dy
	if isFiniteF32(len2) {
		if len2 < 1e-12 {
			return 1.0
		}
		return clampF32(((px-startX)*dx+(py-startY)*dy)/len2, 0.0, 1.0)
	}
	dx64 := float64(endX) - float64(startX)
	dy64 := float64(endY) - float64(startY)
	t := ((float64(px)-float64(startX))*dx64 + (float64(py)-float64(startY))*dy64) / (dx64*dx64 + dy64*dy64)
	return clampF32(float32(t), 0.0, 1.0)
}

// DrawGradient fills a linear or radial gradient in software.
func DrawGradient(target *RgbaTarget, desc *GradientDesc) error {
	return DrawRoundedGradient(target, desc, 0.0)
}

// DrawRoundedGradient fills a linear or radial gradient clipped to an SDF rounded box.
func DrawRoundedGradient(target *RgbaTarget, desc *GradientDesc, cornerRadius float32) error {
	if !isFiniteF32(cornerRadius) || cornerRadius < 0.0 {
		return ErrBadInput
	}
	bounds, stops, ok, err := planGradient(target, desc)
	if err != nil || !ok {
		return err
	}
	corner := maxF32(cornerRadius, 0.0)
	halfW := float32(float64(desc.Rect.Width) * 0.5)
	halfH := float32(float64(desc.Rect.Height) * 0.5)
	cx := float32(float64(desc.Rect.X) + float64(desc.Rect.Width)*0.5)
	cy := float32(float64(desc.Rect.Y) + float64(desc.Rect.Height)*0.5)
	for y := bounds.Y0; y < bounds.Y1; y++ {
		py := float32(y) + 0.5
		for x := bounds.X0; x < bounds.X1; x++ {
			px := float32(x) + 0.5
			cov := float32(1.0)
			if corner > 0.0 {
				d := SDFRoundedBox(px-cx, py-cy, halfW, halfH, corner)
				cov = clampF32(0.5-d, 0.0, 1.0)
			}
			if cov <= 0.0 {
				continue
			}
			var color GpuColor
			if desc.Shape.Kind == GradientShapeLinear {
				t := LinearGradientT(desc.Shape.StartX, desc.Shape.StartY, desc.Shape.EndX, desc.Shape.EndY, px, py)
				color = SampleMultiStopGradient(stops, t)
			} else {
				color = SampleRadialGradient(desc.Shape.CenterX, desc.Shape.CenterY, desc.Shape.RadiusX, desc.Shape.RadiusY, px, py, stops)
			}
			OverStraight(target.pxMut(x, y), color, cov)
		}
	}
	return nil
}

// VideoFrameKind identifies the pixel layout of a VideoFrame.
type VideoFrameKind uint8

const (
	VideoFrameRGBA VideoFrameKind = iota
	VideoFrameNV12
	VideoFrameI420
)

// VideoFrame represents a decoded RGBA8, NV12, or I420 video frame.
type VideoFrame struct {
	Kind     VideoFrameKind
	Width    uint32
	Height   uint32
	Pixels   []byte
	Stride   int
	Y        []byte
	YStride  int
	UV       []byte
	UVStride int
	U        []byte
	UStride  int
	V        []byte
	VStride  int
}

// NewRGBAVideoFrame wraps a straight RGBA8 buffer as a VideoFrame.
func NewRGBAVideoFrame(pixels []byte, width, height uint32, stride int) VideoFrame {
	return VideoFrame{
		Kind:   VideoFrameRGBA,
		Width:  width,
		Height: height,
		Pixels: pixels,
		Stride: stride,
	}
}

// VideoFrameFromNv12 wraps an Nv12Surface as a VideoFrame.
func VideoFrameFromNv12(s *Nv12Surface) VideoFrame {
	return VideoFrame{
		Kind:     VideoFrameNV12,
		Width:    s.Width,
		Height:   s.Height,
		Y:        s.YPlane,
		YStride:  int(s.Width),
		UV:       s.UVPlane,
		UVStride: int(s.Width),
	}
}

// VideoFrameFromI420 wraps an I420Surface as a VideoFrame.
func VideoFrameFromI420(s *I420Surface) VideoFrame {
	cw := int((s.Width + 1) / 2)
	return VideoFrame{
		Kind:    VideoFrameI420,
		Width:   s.Width,
		Height:  s.Height,
		Y:       s.YPlane,
		YStride: int(s.Width),
		U:       s.UPlane,
		UStride: cw,
		V:       s.VPlane,
		VStride: cw,
	}
}

func planeFits(length, stride, row, rows int) bool {
	if stride < row || rows <= 0 {
		return false
	}
	needed := stride*(rows-1) + row
	return needed >= row && length >= needed
}

// Validate checks sizes, strides, and plane lengths of the VideoFrame.
func (f *VideoFrame) Validate() error {
	if f.Width == 0 || f.Height == 0 || f.Width > MaxSide || f.Height > MaxSide {
		return ErrBadFrame
	}
	w := int(f.Width)
	h := int(f.Height)
	cw := (w + 1) / 2
	ch := (h + 1) / 2
	var ok bool
	switch f.Kind {
	case VideoFrameRGBA:
		ok = planeFits(len(f.Pixels), f.Stride, w*4, h)
	case VideoFrameNV12:
		ok = planeFits(len(f.Y), f.YStride, w, h) && planeFits(len(f.UV), f.UVStride, cw*2, ch)
	case VideoFrameI420:
		ok = planeFits(len(f.Y), f.YStride, w, h) &&
			planeFits(len(f.U), f.UStride, cw, ch) &&
			planeFits(len(f.V), f.VStride, cw, ch)
	}
	if !ok {
		return ErrBadFrame
	}
	return nil
}

func (f *VideoFrame) luma(x, y int) float32 {
	if f.Kind == VideoFrameNV12 || f.Kind == VideoFrameI420 {
		return float32(f.Y[y*f.YStride+x])
	}
	return 0.0
}

func (f *VideoFrame) chroma(x, y int) (float32, float32) {
	switch f.Kind {
	case VideoFrameNV12:
		i := y*f.UVStride + x*2
		return float32(f.UV[i]), float32(f.UV[i+1])
	case VideoFrameI420:
		return float32(f.U[y*f.UStride+x]), float32(f.V[y*f.VStride+x])
	default:
		return 128.0, 128.0
	}
}

func (f *VideoFrame) rgba(x, y int) [4]uint8 {
	if f.Kind == VideoFrameRGBA {
		i := y*f.Stride + x*4
		return [4]uint8{f.Pixels[i], f.Pixels[i+1], f.Pixels[i+2], f.Pixels[i+3]}
	}
	return [4]uint8{}
}

// Filter specifies how DrawBlit samples a frame when sizes differ.
type Filter uint8

const (
	FilterNearest Filter = iota
	FilterBilinear
)

// BlitDesc describes where a VideoFrame lands in the target.
type BlitDesc struct {
	X, Y          int32
	Width, Height uint32
	Filter        Filter
	Clip          *GpuRect
}

// NewBlitDesc constructs a nearest-texel BlitDesc.
func NewBlitDesc(x, y int32, width, height uint32) BlitDesc {
	return BlitDesc{
		X:      x,
		Y:      y,
		Width:  width,
		Height: height,
		Filter: FilterNearest,
	}
}

func blitBounds(target *RgbaTarget, frame *VideoFrame, desc *BlitDesc) (Bounds, bool, error) {
	if err := frame.Validate(); err != nil {
		return Bounds{}, false, err
	}
	if desc.Width > MaxSide || desc.Height > MaxSide {
		return Bounds{}, false, ErrBadInput
	}
	area, ok, err := targetBounds(target, desc.Clip)
	if err != nil || !ok {
		return Bounds{}, false, err
	}
	if desc.Width == 0 || desc.Height == 0 {
		return Bounds{}, false, nil
	}
	clampX := func(v int64) uint32 {
		if v < 0 {
			return 0
		}
		if v > int64(target.width) {
			return target.width
		}
		return uint32(v)
	}
	clampY := func(v int64) uint32 {
		if v < 0 {
			return 0
		}
		if v > int64(target.height) {
			return target.height
		}
		return uint32(v)
	}
	b := Bounds{
		X0: clampX(int64(desc.X)),
		Y0: clampY(int64(desc.Y)),
		X1: clampX(int64(desc.X) + int64(desc.Width)),
		Y1: clampY(int64(desc.Y) + int64(desc.Height)),
	}
	if b.X0 >= b.X1 || b.Y0 >= b.Y1 {
		return Bounds{}, false, nil
	}
	res, ok := b.intersect(area)
	return res, ok, nil
}

func nearestIndex(d, src, dst uint64) int {
	idx := ((2*d + 1) * src) / (2 * dst)
	if idx >= src {
		idx = src - 1
	}
	return int(idx)
}

func fixedTaps(d, src, dst uint64, size int) (int, int, uint64) {
	den := 2 * dst
	p := (2*(2*d+1)*src*bilinearOne + den) / (2 * den)
	p += bilinearOne / 2
	w := p % bilinearOne
	i := int64(p/bilinearOne) - 1
	last := int64(size - 1)
	clampIdx := func(v int64) int {
		if v < 0 {
			return 0
		}
		if v > last {
			return int(last)
		}
		return int(v)
	}
	return clampIdx(i), clampIdx(i + 1), w
}

func bt709FixedToRGB(y, u, v int64) [3]uint8 {
	yf := max(y-16*256, int64(0)) * 9539
	uf := u - 128*256
	vf := v - 128*256
	out := func(s int64) uint8 {
		val := (max(s, int64(0)) + (1 << 20)) >> 21
		if val > 255 {
			val = 255
		}
		return uint8(val)
	}
	return [3]uint8{
		out(yf + 14686*vf),
		out(yf - 1747*uf - 4366*vf),
		out(yf + 17305*uf),
	}
}

func bilinearRGBA(texels [4][4]uint8, wx, wy uint64) [4]uint8 {
	one := bilinearOne
	row := func(l, r [4]uint8) ([3]uint64, uint64) {
		al, ar := uint64(l[3]), uint64(r[3])
		var pre [3]uint64
		for c := 0; c < 3; c++ {
			v := uint64(l[c])*al*(one-wx) + uint64(r[c])*ar*wx
			pre[c] = (v + 128) >> 8
		}
		return pre, al*(one-wx) + ar*wx
	}
	top, topA := row(texels[0], texels[1])
	bottom, bottomA := row(texels[2], texels[3])
	alpha := topA*(one-wy) + bottomA*wy
	alpha16 := (alpha + 128) >> 8
	if alpha16 == 0 {
		return [4]uint8{0, 0, 0, 0}
	}
	var out [4]uint8
	for c := 0; c < 3; c++ {
		v := top[c]*(one-wy) + bottom[c]*wy
		ch := (v + alpha16/2) / alpha16
		if ch > 255 {
			ch = 255
		}
		out[c] = uint8(ch)
	}
	aOut := (alpha + (1 << 23)) >> 24
	if aOut > 255 {
		aOut = 255
	}
	out[3] = uint8(aOut)
	return out
}

func bilinearPlane(texels [4]uint8, wx, wy uint64) int64 {
	one := bilinearOne
	k := [4]uint64{
		(one - wx) * (one - wy),
		wx * (one - wy),
		(one - wx) * wy,
		wx * wy,
	}
	var sum uint64
	for i := 0; i < 4; i++ {
		sum += uint64(texels[i]) * k[i]
	}
	return int64((sum + (1 << 15)) >> 16)
}

// DrawBlit scales, converts, and copies frame into target in software.
func DrawBlit(target *RgbaTarget, frame *VideoFrame, desc *BlitDesc) error {
	bounds, ok, err := blitBounds(target, frame, desc)
	if err != nil || !ok {
		return err
	}
	sw, sh := int(frame.Width), int(frame.Height)
	cw, ch := (sw+1)/2, (sh+1)/2
	for y := bounds.Y0; y < bounds.Y1; y++ {
		ly := uint64(int64(y) - int64(desc.Y))
		for x := bounds.X0; x < bounds.X1; x++ {
			lx := uint64(int64(x) - int64(desc.X))
			var out [4]uint8
			if desc.Filter == FilterNearest {
				tx := nearestIndex(lx, uint64(sw), uint64(desc.Width))
				ty := nearestIndex(ly, uint64(sh), uint64(desc.Height))
				if frame.Kind == VideoFrameRGBA {
					out = frame.rgba(tx, ty)
				} else {
					u, v := frame.chroma(tx/2, ty/2)
					rgb := BT709YUVF32ToRGB(frame.luma(tx, ty), u, v)
					out = [4]uint8{rgb[0], rgb[1], rgb[2], 255}
				}
			} else {
				dw, dh := uint64(desc.Width), uint64(desc.Height)
				x0, x1, wx := fixedTaps(lx, uint64(sw), dw, sw)
				y0, y1, wy := fixedTaps(ly, uint64(sh), dh, sh)
				if frame.Kind == VideoFrameRGBA {
					out = bilinearRGBA([4][4]uint8{
						frame.rgba(x0, y0),
						frame.rgba(x1, y0),
						frame.rgba(x0, y1),
						frame.rgba(x1, y1),
					}, wx, wy)
				} else {
					luma := bilinearPlane([4]uint8{
						uint8(frame.luma(x0, y0)),
						uint8(frame.luma(x1, y0)),
						uint8(frame.luma(x0, y1)),
						uint8(frame.luma(x1, y1)),
					}, wx, wy)
					c0, c1, cwx := fixedTaps(lx, uint64(sw), 2*dw, cw)
					r0, r1, cwy := fixedTaps(ly, uint64(sh), 2*dh, ch)
					u00, v00 := frame.chroma(c0, r0)
					u10, v10 := frame.chroma(c1, r0)
					u01, v01 := frame.chroma(c0, r1)
					u11, v11 := frame.chroma(c1, r1)
					u := bilinearPlane([4]uint8{uint8(u00), uint8(u10), uint8(u01), uint8(u11)}, cwx, cwy)
					v := bilinearPlane([4]uint8{uint8(v00), uint8(v10), uint8(v01), uint8(v11)}, cwx, cwy)
					rgb := bt709FixedToRGB(luma, u, v)
					out = [4]uint8{rgb[0], rgb[1], rgb[2], 255}
				}
			}
			copy(target.pxMut(x, y), out[:])
		}
	}
	return nil
}

// DrawSolidRect draws an axis-aligned or SDF rounded rectangle with optional border stroke.
func DrawSolidRect(target *RgbaTarget, rect GpuRect, color GpuColor, cornerRadius, borderWidth float32, borderColor GpuColor, clip *GpuRect) error {
	if !rectIsValid(rect) || !isFiniteF32(cornerRadius) || !isFiniteF32(borderWidth) {
		return ErrBadInput
	}
	area, ok, err := targetBounds(target, clip)
	if err != nil || !ok {
		return err
	}
	fill, ok := rectBounds(rect, target.width, target.height)
	if !ok {
		return nil
	}
	bounds, ok := fill.intersect(area)
	if !ok {
		return nil
	}
	r := maxF32(cornerRadius, 0.0)
	bw := maxF32(borderWidth, 0.0)
	if r == 0.0 && bw == 0.0 {
		for y := bounds.Y0; y < bounds.Y1; y++ {
			for x := bounds.X0; x < bounds.X1; x++ {
				OverStraight(target.pxMut(x, y), color, 1.0)
			}
		}
		return nil
	}
	halfW := rect.Width * 0.5
	halfH := rect.Height * 0.5
	cx := rect.X + halfW
	cy := rect.Y + halfH
	for y := bounds.Y0; y < bounds.Y1; y++ {
		sy := (float32(y) + 0.5) - cy
		for x := bounds.X0; x < bounds.X1; x++ {
			sx := (float32(x) + 0.5) - cx
			dist := SDFRoundedBox(sx, sy, halfW, halfH, r)
			outerCov := clampF32(0.5-dist, 0.0, 1.0)
			if outerCov <= 0.0 {
				continue
			}
			pxColor := color
			if bw > 0.0 {
				innerDist := dist + bw
				borderMix := clampF32(0.5+innerDist, 0.0, 1.0)
				pxColor = LerpColor(color, borderColor, borderMix)
			}
			OverStraight(target.pxMut(x, y), pxColor, outerCov)
		}
	}
	return nil
}

// DrawGlyphCoverage blends an 8-bit coverage glyph bitmap (width * height bytes) into target at (x, y).
func DrawGlyphCoverage(target *RgbaTarget, x, y int32, width, height uint32, coverage []byte, color GpuColor, clip *GpuRect) error {
	if width > MaxSide || height > MaxSide {
		return ErrBadInput
	}
	needed := int(width) * int(height)
	if len(coverage) < needed {
		return ErrBadInput
	}
	area, ok, err := targetBounds(target, clip)
	if err != nil || !ok {
		return err
	}
	if width == 0 || height == 0 || color.A == 0 {
		return nil
	}
	for gy := uint32(0); gy < height; gy++ {
		dstY := int64(y) + int64(gy)
		if dstY < int64(area.Y0) || dstY >= int64(area.Y1) {
			continue
		}
		for gx := uint32(0); gx < width; gx++ {
			dstX := int64(x) + int64(gx)
			if dstX < int64(area.X0) || dstX >= int64(area.X1) {
				continue
			}
			cov := coverage[gy*width+gx]
			if cov == 0 {
				continue
			}
			OverStraight(target.pxMut(uint32(dstX), uint32(dstY)), color, float32(cov)/255.0)
		}
	}
	return nil
}

// SceneOpKind identifies the type of operation in a GpuSceneOp.
type SceneOpKind uint8

const (
	SceneOpClear SceneOpKind = iota
	SceneOpPushClip
	SceneOpPopClip
	SceneOpSolidRect
	SceneOpBoxShadow
	SceneOpGradient
	SceneOpRoundedGradient
	SceneOpBlitRGBA
	SceneOpBlitNV12
	SceneOpBlitI420
	SceneOpGlyphCoverage
)

// GpuSceneOp is one recorded operation inside a GpuSceneBatch.
type GpuSceneOp struct {
	Kind         SceneOpKind
	Rect         GpuRect
	Color        GpuColor
	CornerRadius float32
	BorderWidth  float32
	BorderColor  GpuColor
	Shadow       ShadowDesc
	Shape        GradientShape
	Stops        []GpuGradientStop
	Width        uint32
	Height       uint32
	Stride       int
	Pixels       []byte
	NV12         Nv12Surface
	I420         I420Surface
	Blit         BlitDesc
	GlyphX       int32
	GlyphY       int32
}

// GpuSceneBatch is an ordered command buffer of 2D scene operations.
type GpuSceneBatch struct {
	Ops []GpuSceneOp
}

// NewGpuSceneBatch constructs an empty GpuSceneBatch.
func NewGpuSceneBatch() *GpuSceneBatch {
	return &GpuSceneBatch{}
}

func (b *GpuSceneBatch) Len() int      { return len(b.Ops) }
func (b *GpuSceneBatch) IsEmpty() bool { return len(b.Ops) == 0 }
func (b *GpuSceneBatch) ClearOps()     { b.Ops = b.Ops[:0] }
func (b *GpuSceneBatch) PushClear(c GpuColor) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpClear, Color: c})
}
func (b *GpuSceneBatch) PushClip(r GpuRect) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpPushClip, Rect: r})
}
func (b *GpuSceneBatch) PopClip() {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpPopClip})
}
func (b *GpuSceneBatch) PushSolidRect(r GpuRect, c GpuColor) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpSolidRect, Rect: r, Color: c})
}
func (b *GpuSceneBatch) PushRoundedRect(r GpuRect, c GpuColor, cornerRadius, borderWidth float32, borderColor GpuColor) {
	b.Ops = append(b.Ops, GpuSceneOp{
		Kind:         SceneOpSolidRect,
		Rect:         r,
		Color:        c,
		CornerRadius: cornerRadius,
		BorderWidth:  borderWidth,
		BorderColor:  borderColor,
	})
}
func (b *GpuSceneBatch) PushBoxShadow(desc ShadowDesc) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpBoxShadow, Shadow: desc})
}
func (b *GpuSceneBatch) PushGradient(r GpuRect, shape GradientShape, stops []GpuGradientStop) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpGradient, Rect: r, Shape: shape, Stops: stops})
}
func (b *GpuSceneBatch) PushRoundedGradient(r GpuRect, shape GradientShape, cornerRadius float32, stops []GpuGradientStop) {
	b.Ops = append(b.Ops, GpuSceneOp{
		Kind:         SceneOpRoundedGradient,
		Rect:         r,
		Shape:        shape,
		CornerRadius: cornerRadius,
		Stops:        stops,
	})
}
func (b *GpuSceneBatch) PushBlitRGBA(width, height uint32, stride int, pixels []byte, desc BlitDesc) {
	b.Ops = append(b.Ops, GpuSceneOp{
		Kind:   SceneOpBlitRGBA,
		Width:  width,
		Height: height,
		Stride: stride,
		Pixels: pixels,
		Blit:   desc,
	})
}
func (b *GpuSceneBatch) PushBlitNV12(surf Nv12Surface, desc BlitDesc) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpBlitNV12, NV12: surf, Blit: desc})
}
func (b *GpuSceneBatch) PushBlitI420(surf I420Surface, desc BlitDesc) {
	b.Ops = append(b.Ops, GpuSceneOp{Kind: SceneOpBlitI420, I420: surf, Blit: desc})
}
func (b *GpuSceneBatch) PushGlyphCoverage(x, y int32, width, height uint32, coverage []byte, color GpuColor) {
	b.Ops = append(b.Ops, GpuSceneOp{
		Kind:   SceneOpGlyphCoverage,
		GlyphX: x,
		GlyphY: y,
		Width:  width,
		Height: height,
		Pixels: coverage,
		Color:  color,
	})
}

func intersectGpuRect(a, b GpuRect) GpuRect {
	x0 := maxF32(a.X, b.X)
	y0 := maxF32(a.Y, b.Y)
	x1 := minF32(a.X+a.Width, b.X+b.Width)
	y1 := minF32(a.Y+a.Height, b.Y+b.Height)
	return NewGpuRect(x0, y0, maxF32(x1-x0, 0.0), maxF32(y1-y0, 0.0))
}

func mergeClip(stackTop, opClip *GpuRect) *GpuRect {
	if stackTop != nil && opClip != nil {
		inter := intersectGpuRect(*stackTop, *opClip)
		return &inter
	}
	if stackTop != nil {
		cp := *stackTop
		return &cp
	}
	if opClip != nil {
		cp := *opClip
		return &cp
	}
	return nil
}

// DrawSceneBatch validates and executes a GpuSceneBatch against target.
func DrawSceneBatch(target *RgbaTarget, batch *GpuSceneBatch) error {
	var clips []GpuRect
	for i := range batch.Ops {
		op := &batch.Ops[i]
		var activeClip *GpuRect
		if len(clips) > 0 {
			top := clips[len(clips)-1]
			activeClip = &top
		}
		switch op.Kind {
		case SceneOpClear:
			area, ok, err := targetBounds(target, activeClip)
			if err != nil {
				return err
			}
			if !ok {
				continue
			}
			for y := area.Y0; y < area.Y1; y++ {
				for x := area.X0; x < area.X1; x++ {
					px := target.pxMut(x, y)
					px[0] = op.Color.R
					px[1] = op.Color.G
					px[2] = op.Color.B
					px[3] = op.Color.A
				}
			}
		case SceneOpPushClip:
			if !rectIsValid(op.Rect) {
				return ErrBadInput
			}
			next := op.Rect
			if activeClip != nil {
				next = intersectGpuRect(*activeClip, op.Rect)
			}
			clips = append(clips, next)
		case SceneOpPopClip:
			if len(clips) > 0 {
				clips = clips[:len(clips)-1]
			}
		case SceneOpSolidRect:
			if err := DrawSolidRect(target, op.Rect, op.Color, op.CornerRadius, op.BorderWidth, op.BorderColor, activeClip); err != nil {
				return err
			}
		case SceneOpBoxShadow:
			d := op.Shadow
			d.Clip = mergeClip(activeClip, d.Clip)
			if err := DrawBoxShadow(target, &d); err != nil {
				return err
			}
		case SceneOpGradient:
			d := GradientDesc{Rect: op.Rect, Shape: op.Shape, Stops: op.Stops, Clip: activeClip}
			if err := DrawGradient(target, &d); err != nil {
				return err
			}
		case SceneOpRoundedGradient:
			d := GradientDesc{Rect: op.Rect, Shape: op.Shape, Stops: op.Stops, Clip: activeClip}
			if err := DrawRoundedGradient(target, &d, op.CornerRadius); err != nil {
				return err
			}
		case SceneOpBlitRGBA:
			d := op.Blit
			d.Clip = mergeClip(activeClip, d.Clip)
			vf := NewRGBAVideoFrame(op.Pixels, op.Width, op.Height, op.Stride)
			if err := DrawBlit(target, &vf, &d); err != nil {
				return err
			}
		case SceneOpBlitNV12:
			d := op.Blit
			d.Clip = mergeClip(activeClip, d.Clip)
			vf := VideoFrameFromNv12(&op.NV12)
			if err := DrawBlit(target, &vf, &d); err != nil {
				return err
			}
		case SceneOpBlitI420:
			d := op.Blit
			d.Clip = mergeClip(activeClip, d.Clip)
			vf := VideoFrameFromI420(&op.I420)
			if err := DrawBlit(target, &vf, &d); err != nil {
				return err
			}
		case SceneOpGlyphCoverage:
			if err := DrawGlyphCoverage(target, op.GlyphX, op.GlyphY, op.Width, op.Height, op.Pixels, op.Color, activeClip); err != nil {
				return err
			}
		}
	}
	return nil
}

// BackendKind identifies whether a compositor pass used Software or Hardware.
type BackendKind uint8

const (
	BackendSoftware BackendKind = iota
	BackendHardware
)

// HardwareApi identifies the graphics API probed by ProbeHardware.
type HardwareApi uint8

const (
	HardwareApiNone HardwareApi = iota
	HardwareApiEgl
	HardwareApiDirect3D11
)

// HardwareProbe reports the result of probing the platform graphics API.
type HardwareProbe struct {
	Api            HardwareApi
	DeviceOpened   bool
	ApiVersion     [2]int32
	PassesCompiled bool
	SubmitReady    bool
}

// Reason returns why the hardware path is not used, or empty string when it is ready.
func (p HardwareProbe) Reason() string {
	if p.Api == HardwareApiNone {
		return "no graphics API to probe on this platform"
	}
	if !p.DeviceOpened {
		return "no display or device opened"
	}
	if !p.PassesCompiled {
		return "no device context, or the compositor passes did not compile"
	}
	if !p.SubmitReady {
		return "no device draw path, or its self-test did not match software"
	}
	return ""
}

// SelectBackend selects BackendHardware only when ProbeHardware reports no fallback reason.
func SelectBackend(probe HardwareProbe) BackendKind {
	if probe.Reason() == "" {
		return BackendHardware
	}
	return BackendSoftware
}

// ProbeHardware probes the system graphics environment.
func ProbeHardware() HardwareProbe {
	return HardwareProbe{
		Api:            HardwareApiNone,
		DeviceOpened:   false,
		ApiVersion:     [2]int32{0, 0},
		PassesCompiled: false,
		SubmitReady:    false,
	}
}

// ComposeBackend is the interface implemented by SoftwareBackend and any device backend.
type ComposeBackend interface {
	Kind() BackendKind
	BoxShadow(target *RgbaTarget, desc *ShadowDesc) error
	FillGradient(target *RgbaTarget, desc *GradientDesc) error
	FillRoundedGradient(target *RgbaTarget, desc *GradientDesc, cornerRadius float32) error
	BlitFrame(target *RgbaTarget, frame *VideoFrame, desc *BlitDesc) error
	ExecuteSceneBatch(target *RgbaTarget, batch *GpuSceneBatch) error
}

// SoftwareBackend implements ComposeBackend on the CPU.
type SoftwareBackend struct{}

func (SoftwareBackend) Kind() BackendKind { return BackendSoftware }
func (SoftwareBackend) BoxShadow(target *RgbaTarget, desc *ShadowDesc) error {
	return DrawBoxShadow(target, desc)
}
func (SoftwareBackend) FillGradient(target *RgbaTarget, desc *GradientDesc) error {
	return DrawGradient(target, desc)
}
func (SoftwareBackend) FillRoundedGradient(target *RgbaTarget, desc *GradientDesc, cornerRadius float32) error {
	return DrawRoundedGradient(target, desc, cornerRadius)
}
func (SoftwareBackend) BlitFrame(target *RgbaTarget, frame *VideoFrame, desc *BlitDesc) error {
	return DrawBlit(target, frame, desc)
}
func (SoftwareBackend) ExecuteSceneBatch(target *RgbaTarget, batch *GpuSceneBatch) error {
	return DrawSceneBatch(target, batch)
}

// Compositor executes compositor passes on the active backend with transactional software fallback.
type Compositor struct {
	software  SoftwareBackend
	device    ComposeBackend
	probe     HardwareProbe
	fallbacks uint64
}

// OpenCompositor opens a Compositor using ProbeHardware.
func OpenCompositor() *Compositor {
	return &Compositor{
		probe: ProbeHardware(),
	}
}

// SoftwareCompositor opens a software-only Compositor without running a hardware probe.
func SoftwareCompositor() *Compositor {
	return &Compositor{}
}

// CompositorWithDevice opens a Compositor using device first and SoftwareBackend on any error.
func CompositorWithDevice(device ComposeBackend) *Compositor {
	return &Compositor{device: device}
}

func (c *Compositor) ActiveBackend() BackendKind {
	if c.device != nil {
		return c.device.Kind()
	}
	return BackendSoftware
}

func (c *Compositor) Probe() HardwareProbe { return c.probe }
func (c *Compositor) Fallbacks() uint64    { return c.fallbacks }

func (c *Compositor) runTransactional(target *RgbaTarget, bounds Bounds, op func(ComposeBackend, *RgbaTarget) error) error {
	if c.device != nil {
		saved := target.save(bounds)
		if err := op(c.device, target); err == nil {
			devicePassDrawsCounter.Add(1)
			return nil
		}
		target.restore(bounds, saved)
		c.device = nil
		c.fallbacks++
	}
	return op(c.software, target)
}

func (c *Compositor) BoxShadow(target *RgbaTarget, desc *ShadowDesc) error {
	plan, err := planShadow(target, desc)
	if err != nil || plan == nil {
		return err
	}
	return c.runTransactional(target, plan.out, func(b ComposeBackend, t *RgbaTarget) error {
		return b.BoxShadow(t, desc)
	})
}

func (c *Compositor) FillGradient(target *RgbaTarget, desc *GradientDesc) error {
	bounds, _, ok, err := planGradient(target, desc)
	if err != nil || !ok {
		return err
	}
	return c.runTransactional(target, bounds, func(b ComposeBackend, t *RgbaTarget) error {
		return b.FillGradient(t, desc)
	})
}

func (c *Compositor) FillRoundedGradient(target *RgbaTarget, desc *GradientDesc, cornerRadius float32) error {
	if !isFiniteF32(cornerRadius) || cornerRadius < 0.0 {
		return ErrBadInput
	}
	bounds, _, ok, err := planGradient(target, desc)
	if err != nil || !ok {
		return err
	}
	return c.runTransactional(target, bounds, func(b ComposeBackend, t *RgbaTarget) error {
		return b.FillRoundedGradient(t, desc, cornerRadius)
	})
}

func (c *Compositor) BlitFrame(target *RgbaTarget, frame *VideoFrame, desc *BlitDesc) error {
	bounds, ok, err := blitBounds(target, frame, desc)
	if err != nil || !ok {
		return err
	}
	return c.runTransactional(target, bounds, func(b ComposeBackend, t *RgbaTarget) error {
		return b.BlitFrame(t, frame, desc)
	})
}

func (c *Compositor) ExecuteSceneBatch(target *RgbaTarget, batch *GpuSceneBatch) error {
	if batch.IsEmpty() || target.Width() == 0 || target.Height() == 0 {
		return nil
	}
	full := Bounds{X0: 0, Y0: 0, X1: target.Width(), Y1: target.Height()}
	err := c.runTransactional(target, full, func(b ComposeBackend, t *RgbaTarget) error {
		return b.ExecuteSceneBatch(t, batch)
	})
	if err == nil {
		recordSceneBatchTelemetry(batch.Len())
	}
	return err
}
