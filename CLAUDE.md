# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- `npm run dev` - Start development server (runs on http://localhost:5173)
- `npm run build` - Build production version
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint to check code quality

## Project Architecture

This is a React + Three.js portfolio website built with Vite. The site showcases an interactive 3D experience with sections for hero, about, projects, publications, work experience, and contact.

### Key Technologies
- **React 18** - UI framework
- **Three.js + React Three Fiber** - 3D graphics and animations
- **React Three Drei** - Helper utilities for Three.js
- **GSAP** - Animations and transitions
- **Tailwind CSS** - Styling framework
- **EmailJS** - Contact form functionality
- **Leva** - 3D development controls (hidden in production)

### Core Structure

**Main App Flow**: `App.jsx` renders sections in order: Navbar → Hero → About → Projects → Publications → WorkExperience → Contact → Footer

**3D Scene Architecture**:
- `Hero.jsx` contains the main 3D canvas with interactive elements
- `HeroCamera.jsx` handles camera controls and mouse interactions
- `HackerRoom.jsx` is the main 3D room model
- Floating 3D elements: `Cube.jsx`, `Rings.jsx`, tech logos (`PythonLogo.jsx`, `PyTorchLogo.jsx`, etc.)
- `calculateSizes()` in `constants/index.js` handles responsive positioning

**Data Management**:
- `src/constants/index.js` contains all static data (projects, publications, work experience, navigation)
- Responsive breakpoints handled via `react-responsive` hooks

**Component Pattern**:
- 3D components in `/components` directory
- Page sections in `/sections` directory
- Shared utilities in `/hooks` directory
- All 3D models stored in `/public/models`
- Textures and assets in `/public/textures` and `/public/assets`

### Email Configuration

The contact form uses EmailJS. Environment variables needed:
- `REACT_APP_EMAILJS_USERID`
- `REACT_APP_EMAILJS_TEMPLATEID`
- `REACT_APP_EMAILJS_RECEIVERID`

### 3D Model Loading

Models are loaded from `/public/models` using `useGLTF` hook. Key models:
- `hacker-room.glb` - Main desk/room scene
- `computer.glb` - Interactive computer for projects
- `cube.glb`, `react.glb` - Floating elements
- Animation files in `/models/animations/`

### Performance Considerations

- All 3D models are preloaded using `useGLTF.preload()`
- Responsive sizing calculated once and passed to components
- Suspense boundaries with custom loading components
- Media queries determine render complexity based on device