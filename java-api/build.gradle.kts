plugins {
    `java-library`
    `maven-publish`
}

group = "com.frc6324"
version = "0.1.0"

repositories {
    // Use Maven Central for resolving dependencies.
    mavenCentral()
}

dependencies {
    // No runtime dependencies: the FFM bindings only need java.base.
}

testing {
    suites {
        // Configure the built-in test suite
        val test = named<JvmTestSuite>("test") {
            // Use JUnit Jupiter test framework
            useJUnitJupiter("6.0.1")
        }
    }
}

// Apply a specific Java toolchain to ease working on different environments.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
    // Published alongside the binary so consumers get the module sources and
    // generated API docs from the local Maven repository too.
    withSourcesJar()
    withJavadocJar()
}

// ---------------------------------------------------------------------------
// Publishing
//
// `./gradlew publishToMavenLocal` installs
// com.frc6324:bdcudatag:<version> under ~/.m2/repository, so sibling builds
// can depend on it with `mavenLocal()` instead of a project path.
// ---------------------------------------------------------------------------
publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            groupId = "com.frc6324"
            artifactId = "bdcudatag"
            version = project.version.toString()

            pom {
                name = "bdcudatag"
                description = "CUDA AprilTag-family detector (tag36h11, tag16h5) with Java FFM bindings."
                url = "https://github.com/cobouch/bd-apriltag"

                licenses {
                    license {
                        name = "BSD 3-Clause License"
                        url = "https://opensource.org/license/bsd-3-clause/"
                        distribution = "repo"
                    }
                }

                scm {
                    connection = "scm:git:git://github.com/cobouch/bd-apriltag.git"
                    developerConnection = "scm:git:ssh://github.com/cobouch/bd-apriltag.git"
                    url = "https://github.com/cobouch/bd-apriltag"
                }
            }
        }
    }

    repositories {
        mavenLocal()
    }
}

// ---------------------------------------------------------------------------
// Bundled native library
//
// The detector's shared library is copied into the jar under
// com/frc6324/bdcudatag/internal/native/<platform>/. NativeLibrary extracts it
// to a temporary file on first use, so consumers need nothing but the jar and
// no particular location for detector/ on disk.
// ---------------------------------------------------------------------------
val nativeLibraryName: String = System.mapLibraryName("bdcudatag")

val nativePlatform: String = run {
    val osName = System.getProperty("os.name").lowercase()
    val os = when {
        osName.contains("win") -> "windows"
        osName.contains("mac") || osName.contains("darwin") -> "macos"
        else -> "linux"
    }
    val rawArch = System.getProperty("os.arch").lowercase()
    val arch = when (rawArch) {
        "amd64", "x86_64", "x64" -> "x86_64"
        "aarch64", "arm64" -> "aarch64"
        else -> rawArch.filter { it.isLetterOrDigit() || it == '_' }
    }
    "$os-$arch"
}

/** The native library to package, or null when detector/ has not been built. */
val nativeLibrarySource: java.io.File? = run {
    val configured = providers.gradleProperty("bdcudatag.native.dir").orNull
    val roots = buildList {
        configured?.let { add(file(it)) }
        add(layout.projectDirectory.dir("../detector/build/Release").asFile)
        add(layout.projectDirectory.dir("../detector/build").asFile)
        add(layout.projectDirectory.dir("../detector/build/lib").asFile)
        add(layout.projectDirectory.dir("native/$nativePlatform").asFile)
    }
    roots.map { root -> if (root.isFile) root else root.resolve(nativeLibraryName) }
        .firstOrNull { it.isFile }
}

val generatedNativeResources = layout.buildDirectory.dir("generated/native-resources")

val bundleNativeLibrary = tasks.register<Sync>("bundleNativeLibrary") {
    group = "build"
    description = "Copies the bdcudatag native library into the jar as a resource."
    into(generatedNativeResources)
    if (nativeLibrarySource != null) {
        from(nativeLibrarySource) {
            into("com/frc6324/bdcudatag/internal/native/$nativePlatform")
        }
    } else {
        doFirst {
            logger.warn(
                "bdcudatag: no $nativeLibraryName found, so the jar will not bundle a native " +
                    "library. Build detector/ first, or point -Pbdcudatag.native.dir=<directory> " +
                    "at an existing build."
            )
        }
    }
}

// The staged native library goes into the binary jar only, never the resource
// output or sources jar (which is what keeps `publishToMavenLocal` consistent).
tasks.jar {
    dependsOn(bundleNativeLibrary)
    from(generatedNativeResources)
}

tasks.test {
    // Tests load the native library from the unnamed module (classpath).
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
