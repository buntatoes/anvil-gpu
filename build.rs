// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
fn main() {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
