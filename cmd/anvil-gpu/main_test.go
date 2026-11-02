// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"

	anvilgpu "github.com/buntatoes/anvil-gpu"
)

func TestWorkerRejectsOversizedDimensionsAndBuffers(t *testing.T) {
	var req, resp bytes.Buffer
	var hdr [16]byte

	// opRGBAToNV12 with width > MaxSide must fail with ErrBadTarget without allocating.
	req.WriteByte(opRGBAToNV12)
	binary.LittleEndian.PutUint32(hdr[0:4], anvilgpu.MaxSide+2)
	binary.LittleEndian.PutUint32(hdr[4:8], 64)
	req.Write(hdr[:8])
	if err := runWorker(&req, &resp); !errors.Is(err, anvilgpu.ErrBadTarget) {
		t.Fatalf("expected ErrBadTarget on oversized opRGBAToNV12 width, got %v", err)
	}

	// opNV12ToRGBA with yLen = 0xFFFFFFFF must fail with ErrBadTarget without allocating.
	req.Reset()
	resp.Reset()
	req.WriteByte(opNV12ToRGBA)
	binary.LittleEndian.PutUint32(hdr[0:4], 16)
	binary.LittleEndian.PutUint32(hdr[4:8], 16)
	binary.LittleEndian.PutUint32(hdr[8:12], 0xFFFFFFFF)
	binary.LittleEndian.PutUint32(hdr[12:16], 128)
	req.Write(hdr[:16])
	if err := runWorker(&req, &resp); !errors.Is(err, anvilgpu.ErrBadTarget) {
		t.Fatalf("expected ErrBadTarget on oversized opNV12ToRGBA yLen, got %v", err)
	}
}
