plugins {
    application
}

group = "com.frc6324"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    // Resolved from the ../java-api composite build, including the bundled native library.
    implementation("com.frc6324:bdcudatag:0.1.0")
    // Camera capture and image decoding. Ships Windows x86_64 and Linux x86_64/ARMv8 natives.
    implementation("org.openpnp:opencv:4.9.0-0")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "com.frc6324.bdcudatag.viewer.CameraViewer"
    // The detector uses the FFM API, which is a restricted operation on Java 25.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

tasks.named<JavaExec>("run") {
    // Forward an explicit native library location to the application JVM.
    providers.systemProperty("bdcudatag.library.path").orNull?.let {
        systemProperty("bdcudatag.library.path", it)
    }
}
