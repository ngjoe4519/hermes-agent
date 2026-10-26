// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

// Simulate Linux: pin the platform AND tell the preload bridge that glass is
// unsupported there (`glassSupportedOn` is macOS, plus Windows 11 22H2+).
// Both must be in place BEFORE the store module is evaluated (hoisted above
// the imports) so GLASS_SUPPORTED resolves false.
//
// Linux has exactly ONE half of the setting: Clear, via `setOpacity`, which
// Chromium's Ozone X11/Wayland backends honour. It used to have none — Linux
// was excluded from `translucencySupportedOn`, so Settings hid the row while
// the RESOLVED value still came from the mac glass table: intensity 29 in clear
// mode is `opacityRamp(29) = 0.94113`, so every fresh Linux profile faded its
// whole window, text included, with no control anywhere to undo it. Now Linux
// gets the Clear slider (the same single-lever row Windows 10 renders) and an
// untouched profile starts fully opaque.
vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'Linux x86_64' })
  Object.defineProperty(globalThis.window, 'hermesDesktop', {
    configurable: true,
    value: { glassSupported: false, translucencySupported: true }
  })
})

import {
  $translucency,
  $translucencyBook,
  GLASS_SUPPORTED,
  setAppearance,
  setTranslucency,
  TRANSLUCENCY_SUPPORTED
} from './translucency'

const clearAttr = () => document.documentElement.hasAttribute('data-hermes-clear')

describe('translucency on Linux (Clear only)', () => {
  it('runs its assertions for real in this environment', () => {
    // Guard the guard: if the platform pin or the bridge stub ever stops
    // landing, every assertion below silently tests another platform.
    expect(navigator.platform).toBe('Linux x86_64')
    expect(TRANSLUCENCY_SUPPORTED).toBe(true)
    expect(GLASS_SUPPORTED).toBe(false)
  })

  it('starts fully opaque, not at the mac glass tint', () => {
    expect($translucency.get()).toMatchObject({ mode: 'clear', intensity: 0, fade: 0 })

    setAppearance('light')
    expect($translucency.get().intensity).toBe(0)
    setAppearance('dark')
  })

  // The upgrade path. The pre-fix build never let a Linux profile EDIT this —
  // no row, so the renderer's book stayed empty and only main's
  // `translucency.json` carried the leaked 29 (which the renderer corrects on
  // its first echo). An empty book must therefore resolve the new default.
  it('resolves the pre-fix book (no user value) to the clear-mode default', () => {
    $translucencyBook.set({ mode: 'clear', base: {}, light: {}, dark: {} })

    expect($translucency.get().intensity).toBe(0)
    expect(clearAttr()).toBe(false)
  })

  // ...and nothing here overrides a value that really was persisted.
  it('honours a persisted clear value', () => {
    $translucencyBook.set({ mode: 'clear', base: { intensity: 40 }, light: {}, dark: {} })

    expect($translucency.get().intensity).toBe(40)
  })

  // ...but the lever is live now, so raising it must actually fade the window.
  it('fades the window when the user raises the Clear lever', () => {
    setTranslucency(40)

    expect($translucency.get()).toMatchObject({ mode: 'clear', intensity: 40 })
    expect(clearAttr()).toBe(true)

    setTranslucency(0)
    expect(clearAttr()).toBe(false)
  })
})
