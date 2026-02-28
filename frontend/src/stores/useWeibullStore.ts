/**
 * 功率模块寿命分析软件 - 威布尔可靠性分析状态管理
 * @author GSH
 *
 * Zustand store for the Weibull reliability analysis page.
 * State is kept in memory and persists within SPA navigation.
 */
import { create } from 'zustand'
import type {
  WeibullFitResult,
  WeibullProbabilityPlotResult,
  WeibullCurveResult,
} from '@/types'

/* ---------- Types ---------- */
export interface WeibullPageState {
  /* UI */
  tab: number
  loading: boolean
  error: string | null

  /* Tab 0 — 数据输入 */
  failureTimesInput: string
  censoredTimesInput: string
  confidenceLevel: string
  fitMethod: 'mle' | 'ls'

  /* 拟合结果 */
  fitResult: WeibullFitResult | null

  /* 概率图数据 */
  probabilityPlotData: WeibullProbabilityPlotResult | null

  /* 可靠度曲线数据 */
  reliabilityCurveData: WeibullCurveResult | null

  /* 失效率曲线数据 */
  hazardCurveData: WeibullCurveResult | null

  /* 图表选项 */
  timeRangeMin: string
  timeRangeMax: string
  showConfidenceInterval: boolean

  /* 自定义B寿命 */
  customBLifes: number[]
  customBLifeResults: Record<number, number>
}

export interface WeibullPageActions {
  /** Patch one or more fields (shallow merge). */
  patch: (partial: Partial<WeibullPageState>) => void
  /** Reset everything to initial defaults. */
  reset: () => void
}

/* ---------- Defaults ---------- */
const initialState: WeibullPageState = {
  tab: 0,
  loading: false,
  error: null,

  failureTimesInput: '',
  censoredTimesInput: '',
  confidenceLevel: '0.9',
  fitMethod: 'ls',  // 默认最小二乘法

  fitResult: null,
  probabilityPlotData: null,
  reliabilityCurveData: null,
  hazardCurveData: null,

  timeRangeMin: '',
  timeRangeMax: '',
  showConfidenceInterval: true,

  customBLifes: [],
  customBLifeResults: {},
}

/* ---------- Store ---------- */
export const useWeibullStore = create<WeibullPageState & WeibullPageActions>()(
  (set) => ({
    ...initialState,
    patch: (partial) => set(partial),
    reset: () => set(initialState),
  })
)

export default useWeibullStore
