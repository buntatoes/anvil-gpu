// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
fn main() {
    // Ask about the target, not the machine doing the build, so a Windows
    // build made on Linux does not try to link libdl.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
