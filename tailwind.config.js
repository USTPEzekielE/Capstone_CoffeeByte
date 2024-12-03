/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['index.js', 'App.tsx', 'src/**/*.{ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        primary: '#3E735B',
      },
    },
  },
  plugins: [],
};
