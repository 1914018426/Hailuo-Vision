/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        teal: {
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
        },
        slate: {
          750: '#1e293b',
          850: '#0f172a',
          950: '#020617',
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      animation: {
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'pulse-ring': 'pulseRing 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
        'border-pulse': 'borderPulse 2s ease-in-out infinite',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 5px rgba(45, 212, 191, 0.3)' },
          '50%': { boxShadow: '0 0 20px rgba(45, 212, 191, 0.6)' },
        },
        pulseRing: {
          '0%': { transform: 'scale(0.8)', opacity: '1' },
          '100%': { transform: 'scale(2.4)', opacity: '0' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        borderPulse: {
          '0%, 100%': { borderColor: 'rgba(255, 255, 255, 0.2)' },
          '50%': { borderColor: 'rgba(255, 255, 255, 0.6)' },
        },
      },
      boxShadow: {
        'glow': '0 0 15px rgba(45, 212, 191, 0.3)',
        'glow-lg': '0 0 30px rgba(45, 212, 191, 0.4)',
        'glow-red': '0 0 15px rgba(239, 68, 68, 0.3)',
        'glow-slate': '0 0 15px rgba(148, 163, 184, 0.3)',
        'glow-orange': '0 0 15px rgba(249, 115, 22, 0.3)',
      },
    },
  },
  plugins: [],
};
