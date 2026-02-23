platform :ios, '16.0'

# Disable input/output paths to work around Xcode caching
install! 'cocoapods', :disable_input_output_paths => true

target 'SileroTTS' do
  use_frameworks!
  pod 'LibTorch', '~> 1.13.0'
end

target 'SileroTTSExtension' do
  use_frameworks!
  pod 'LibTorch', '~> 1.13.0'
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '16.0'
      config.build_settings['EXCLUDED_ARCHS[sdk=iphonesimulator*]'] = 'arm64'
    end
  end
end
