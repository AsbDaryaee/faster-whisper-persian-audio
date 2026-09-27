# Transcribe Persian Audio to Markdown

## Overview

This project provides a tool to transcribe Persian language audio or video lectures into **Markdown** files. It utilizes the **fastest-whisper** model for balanced speed and accuracy in Farsi speech recognition.

## Usage Examples

### Run the script

```bash
py -u transcribe_lecture.py path/to/file.mp3 --model tiny   # Default model
py -u transcribe_lecture.py path/to/file.mp3 --preview 180   # Test for first 3 minutes
py -u transcribe_lecture.py path/to/file.mp3 --verbose      # Show each word with verbose output
```

## Features

- Transcribes Persian audio/video to Markdown
- Supports custom model sizes (`tiny`, `base`, `small`, `medium`, `large-v3`)
- Optional preview for short segments (e.g., first 180 seconds)
- Detailed progress and transcript feedback
- Outputs a markdown file with the full transcription

## Requirements

- Install required dependencies:
  ```
  pip install fastwhisper
  ```
