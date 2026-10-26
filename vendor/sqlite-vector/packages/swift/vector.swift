// vector.swift
// Provides the path to the vector SQLite extension for use with sqlite3_load_extension.

import Foundation

public struct vector {
    /// Returns the absolute path to the vector dylib for use with sqlite3_load_extension.
    public static var path: String {
        #if os(macOS)
        return Bundle.main.bundlePath + "/Contents/Frameworks/vector.framework/vector"
        #else
        return Bundle.main.bundlePath + "/Frameworks/vector.framework/vector"
        #endif
    }
}
