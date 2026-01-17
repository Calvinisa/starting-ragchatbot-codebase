# Frontend Changes: Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

## Files Modified

### 1. `frontend/index.html`
- Added theme toggle button with sun/moon SVG icons positioned at the top-right corner
- Button includes proper accessibility attributes (`aria-label`, `title`)

### 2. `frontend/style.css`

#### New CSS Variables for Light Theme
Added `[data-theme="light"]` selector with light theme color palette:
- `--background: #f8fafc` - Light gray background
- `--surface: #ffffff` - White surface color
- `--surface-hover: #f1f5f9` - Light hover state
- `--text-primary: #1e293b` - Dark text for contrast
- `--text-secondary: #64748b` - Muted secondary text
- `--border-color: #e2e8f0` - Light borders
- `--assistant-message: #f1f5f9` - Light message bubbles
- `--code-bg: rgba(0, 0, 0, 0.05)` - Subtle code backgrounds

#### Theme Toggle Button Styles
- Fixed positioning at top-right (1rem from edges)
- Circular button (44x44px) with surface background
- Hover effects: scale, color change, border highlight
- Focus state with ring for keyboard accessibility
- Icon switching: sun icon visible in dark mode, moon icon in light mode

#### Smooth Transitions
Added `transition` properties to key elements for smooth theme switching:
- `background-color`, `color`, and `border-color` transitions (0.3s ease)
- Applied to body, container, sidebar, chat areas, inputs, and buttons

### 3. `frontend/script.js`

#### New Functions
- `initializeTheme()`: Loads saved theme from localStorage on page load (defaults to dark)
- `toggleTheme()`: Switches between themes and persists preference to localStorage

#### Event Listener
- Added click handler for theme toggle button

## Features
- Persistent theme preference using localStorage
- Smooth 0.3s transitions between themes
- Keyboard accessible (Tab + Enter/Space to toggle)
- Mobile-friendly circular button design
- Icons switch based on current theme (sun = dark mode, moon = light mode)

## Accessibility
- Button has `aria-label="Toggle dark/light theme"` for screen readers
- Focus ring visible when navigating with keyboard
- Sufficient color contrast maintained in both themes
