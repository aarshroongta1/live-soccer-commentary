#!/usr/bin/env bash
# Prints the avfoundation device indices. Pick "<screen>:<audio>" for AVFOUNDATION_DEVICE.
# Audio capture of system output needs a loopback device (BlackHole or Loopback).
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | sed -n '/AVFoundation/,$p'
