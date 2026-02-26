import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import type {
  LifetimeModelType,
  PredictionResult,
  ExperimentData,
  RainflowResult,
  DamageAccumulationResult,
  WeibullAnalysisResult,
  SensitivityAnalysisResult,
} from '@/types'

// ============================================
// Application State Interface
// ============================================

interface AppState {
  // UI State
  sidebarOpen: boolean
  currentPath: string
  themeMode: 'light' | 'dark'
  language: 'zh' | 'en'

  // Prediction State
  selectedModel: LifetimeModelType
  predictionResult: PredictionResult | null
  predictionHistory: PredictionResult[]

  // Data State
  experimentData: ExperimentData[]
  selectedExperimentId: string | null

  // Rainflow State
  rainflowResult: RainflowResult | null

  // Damage State
  damageResult: DamageAccumulationResult | null

  // Analysis State
  weibullResult: WeibullAnalysisResult | null
  sensitivityResult: SensitivityAnalysisResult | null

  // Loading State
  isLoading: boolean
  error: string | null

  // Actions
  setSidebarOpen: (open: boolean) => void
  setCurrentPath: (path: string) => void
  setThemeMode: (mode: 'light' | 'dark') => void
  setLanguage: (lang: 'zh' | 'en') => void

  setSelectedModel: (model: LifetimeModelType) => void
  setPredictionResult: (result: PredictionResult | null) => void
  addPredictionToHistory: (result: PredictionResult) => void
  clearPredictionHistory: () => void

  setExperimentData: (data: ExperimentData[]) => void
  addExperimentData: (data: ExperimentData) => void
  removeExperimentData: (id: string) => void
  setSelectedExperimentId: (id: string | null) => void

  setRainflowResult: (result: RainflowResult | null) => void

  setDamageResult: (result: DamageAccumulationResult | null) => void

  setWeibullResult: (result: WeibullAnalysisResult | null) => void
  setSensitivityResult: (result: SensitivityAnalysisResult | null) => void

  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // Reset functions
  resetPredictionState: () => void
  resetAnalysisState: () => void
  resetAll: () => void
}

// ============================================
// Initial State
// ============================================

const initialState = {
  sidebarOpen: true,
  currentPath: '/',
  themeMode: 'light' as const,
  language: 'zh' as const,
  selectedModel: 'cips2008' as LifetimeModelType,
  predictionResult: null,
  predictionHistory: [],
  experimentData: [],
  selectedExperimentId: null,
  rainflowResult: null,
  damageResult: null,
  weibullResult: null,
  sensitivityResult: null,
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
        setLanguage: (lang) => set({ language: lang }),

        // Prediction Actions
        setSelectedModel: (model) => set({ selectedModel: model }),
        setPredictionResult: (result) => set({ predictionResult: result }),
        addPredictionToHistory: (result) =>
          set((state) => ({
            predictionHistory: [result, ...state.predictionHistory].slice(0, 50), // Keep last 50
          })),
        clearPredictionHistory: () => set({ predictionHistory: [] }),

        // Experiment Data Actions
        setExperimentData: (data) => set({ experimentData: data }),
        addExperimentData: (data) =>
          set((state) => ({
            experimentData: [...state.experimentData, data],
          })),
        removeExperimentData: (id) =>
          set((state) => ({
            experimentData: state.experimentData.filter((d) => d.id !== id),
            selectedExperimentId:
              state.selectedExperimentId === id ? null : state.selectedExperimentId,
          })),
        setSelectedExperimentId: (id) => set({ selectedExperimentId: id }),

        // Rainflow Actions
        setRainflowResult: (result) => set({ rainflowResult: result }),

        // Damage Actions
        setDamageResult: (result) => set({ damageResult: result }),

        // Analysis Actions
        setWeibullResult: (result) => set({ weibullResult: result }),
        setSensitivityResult: (result) => set({ sensitivityResult: result }),

        // Loading & Error Actions
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),

        // Reset functions
        resetPredictionState: () =>
          set({
            predictionResult: null,
            rainflowResult: null,
            damageResult: null,
          }),

        resetAnalysisState: () =>
          set({
            weibullResult: null,
            sensitivityResult: null,
          }),

        resetAll: () => set(initialState),
      }),
      {
        name: 'cips2008-storage',
        partialize: (state) => ({
          themeMode: state.themeMode,
          language: state.language,
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

export const usePredictionState = () =>
  useStore((state) => ({
    selectedModel: state.selectedModel,
    predictionResult: state.predictionResult,
    predictionHistory: state.predictionHistory,
    setSelectedModel: state.setSelectedModel,
    setPredictionResult: state.setPredictionResult,
    addPredictionToHistory: state.addPredictionToHistory,
    clearPredictionHistory: state.clearPredictionHistory,
  }))

export const useExperimentState = () =>
  useStore((state) => ({
    experimentData: state.experimentData,
    selectedExperimentId: state.selectedExperimentId,
    setExperimentData: state.setExperimentData,
    addExperimentData: state.addExperimentData,
    removeExperimentData: state.removeExperimentData,
    setSelectedExperimentId: state.setSelectedExperimentId,
  }))

export const useAnalysisState = () =>
  useStore((state) => ({
    rainflowResult: state.rainflowResult,
    damageResult: state.damageResult,
    weibullResult: state.weibullResult,
    sensitivityResult: state.sensitivityResult,
    setRainflowResult: state.setRainflowResult,
    setDamageResult: state.setDamageResult,
    setWeibullResult: state.setWeibullResult,
    setSensitivityResult: state.setSensitivityResult,
  }))

export const useUIState = () =>
  useStore((state) => ({
    sidebarOpen: state.sidebarOpen,
    currentPath: state.currentPath,
    themeMode: state.themeMode,
    language: state.language,
    setSidebarOpen: state.setSidebarOpen,
    setCurrentPath: state.setCurrentPath,
    setThemeMode: state.setThemeMode,
    setLanguage: state.setLanguage,
  }))

export const useLoadingState = () =>
  useStore((state) => ({
    isLoading: state.isLoading,
    error: state.error,
    setLoading: state.setLoading,
    setError: state.setError,
  }))

export default useStore
