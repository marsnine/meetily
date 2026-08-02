#!/bin/bash
# Build MeetilyEdge for the iPhone, fix the nested-dylib code signature that
# Xcode/SPM leaves invalid inside CLiteRTLM.framework, and install to the device.
#
# Why the re-sign: the community LiteRTLM-Swift xcframework ships
# libGemmaModelConstraintProvider.dylib *inside* CLiteRTLM.framework. Xcode embeds
# it but does NOT re-sign it with our dev cert, so dyld rejects it at launch
# ("code signature invalid"). We re-sign dylib -> framework -> app after the build.
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

DEV="${DEVICE_UDID:-7AE584D7-A7B7-57BE-A284-1C82ED466A9E}"
TEAM="${APPLE_TEAM:-PC4J95QKMH}"
APP="build/Build/Products/Debug-iphoneos/MeetilyEdge.app"
FW="$APP/Frameworks/CLiteRTLM.framework"

echo "==> xcodegen"
xcodegen generate >/dev/null

echo "==> xcodebuild (device, Debug, automatic signing)"
xcodebuild -project MeetilyEdge.xcodeproj -scheme MeetilyEdge -configuration Debug \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$TEAM" build \
  | tail -3

ID=$(security find-identity -v -p codesigning | grep -i "marsnine" | head -1 | grep -oE '"[^"]+"' | tr -d '"')
echo "==> re-sign runtime dylibs + framework with: $ID"
codesign -d --entitlements :/tmp/edge_ent.plist "$APP"
# Official LiteRT-LM runtime/accelerator dylibs live at app/Frameworks/*.dylib
for dylib in "$APP"/Frameworks/*.dylib; do
  echo "   sign $(basename "$dylib")"; codesign -f -s "$ID" "$dylib"
done
codesign -f -s "$ID" "$FW"
codesign -f -s "$ID" --entitlements /tmp/edge_ent.plist "$APP"
codesign -v --strict "$APP" && echo "   signature OK"

echo "==> install to device $DEV"
xcrun devicectl device install app --device "$DEV" "$APP" | grep -iE "App installed|bundleID|error"
echo "==> done"
