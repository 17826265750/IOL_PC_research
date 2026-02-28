import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import type {
  LifetimeModelType,
  PredictionResult,
  RainflowResult,
} from '@/types'

// ============================================
// Application State Interface
// ============================================

interface AppState {
  // UI State
  sidebarOpen: boolean
  currentPath: string
  themeMode: 'light' | 'dark'

  // Prediction State
  selectedModel: LifetimeModelType
  predictionResult: PredictionResult | null
  predictionHistory: PredictionResult[]

  // Rainflow State
  rainflowResult: RainflowResult | null

  // Loading State
  isLoading: boolean
  error: string | null

  // Actions
  setSidebarOpen: (open: boolean) => void
  setCurrentPath: (path: string) => void
  setThemeMode: (mode: 'light' | 'dark') => void

  setSelectedModel: (model: LifetimeModelType) => void
  setPredictionResult: (result: PredictionResult | null) => void
  addPredictionToHistory: (result: PredictionResult) => void
  clearPredictionHistory: () => void

  setRainflowResult: (result: RainflowResult | null) => void

  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // Reset functions
  resetPredictionState: () => void
  resetAll: () => void
}

// ============================================
// Initial State
// ============================================

const initialState = {
  sidebarOpen: true,
  currentPath: '/',
  themeMode: 'light' as const,
  selectedModel: 'cips2008' as LifetimeModelType,
  predictionResult: null,
  predictionHistory: [],
  rainflowResult: null,
  isLoading: false,
  error: null,
}

// ============================================
// Create Store
// ============================================

export const useStore = create<AppState>()(
  devtools(
    persist(
      (set) => ({
        ...initialState,

        // UI Actions
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        setCurrentPath: (path) => set({ currentPath: path }),
        setThemeMode: (mode) => set({ themeMode: mode }),

        // Prediction Actions
        setSelectedModel: (model) => set({ selectedModel: model }),
        setPredictionResult: (result) => set({ predictionResult: result }),
        addPredictionToHistory: (result) =>
          set((state) => ({
            predictionHistory: [result, ...state.predictionHistory].slice(0, 50),
          })),
        clearPredictionHistory: () => set({ predictionHistory: [] }),

        // Rainflow Actions
        setRainflowResult: (result) => set({ rainflowResult: result }),

        // Loading & Error Actions
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),

        // Reset functions
        resetPredictionState: () =>
          set({
            predictionResult: null,
            rainflowResult: null,
          }),

        resetAll: () => set(initialState),
      }),
      {
        name: 'cips2008-storage',
        partialize: (state) => ({
          themeMode: state.themeMode,
          selectedModel: state.selectedModel,
          predictionHistory: state.predictionHistory,
        }),
      }
    ),
    { name: 'CIPS2008-Store' }
  )
)

// ============================================
// Selector Hooks
// ============================================

export const useUIState = () =>
  useStore((state) => ({
    sidebarOpen: state.sidebarOpen,
    currentPath: state.currentPath,
    themeMode: state.themeMode,
    setSidebarOpen: state.setSidebarOpen,
    setCurrentPath: state.setCurrentPath,
    setThemeMode: state.setThemeMode,
  }))

export const useLoadingState = () =>
  useStore((state) => ({
    isLoading: state.isLoading,
    error: state.error,
    setLoading: state.setLoading,
    setError: state.setError,
  }))

export default useStore
