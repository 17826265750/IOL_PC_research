/**
 * Zustand store for the Rainflow (功率循环寿命评估) page.
 *
 * Keeps all input parameters *and* computed results in a global store so
 * that navigating away from the page and coming back preserves everything.
 */
import { create } from 'zustand'
import type { RainflowResult } from '@/types'

/* ---------- Types ---------- */
export type InputMode = 'power' | 'tj'
export type ZthMode = 'foster' | 'sampled'
export type DamageMethod = 'life_curve' | 'model'

export interface RainflowPageState {
  /* UI */
  tab: number
  loading: boolean
  error: string | null

  /* Tab 0 — 功耗→温度 (single source) */
  inputMode: InputMode
  zthMode: ZthMode
  fosterInput: string
  zthInput: string
  dt: string
  ambientTemp: string

  /* Tab 0 — multi-source */
  sourceCount: number
  sourceNames: string[]
  sourcePowerData: string[]
  zthMatrixInput: string[][] // [node][source] Foster RC text
  targetNode: number

  /* Tab 1 — 雨流参数 */
  nBand: string
  yMin: string
  yMax: string
  ignoreBelow: string
  rearrange: boolean

  /* Tab 2 — 寿命计算 */
  damageMethod: DamageMethod
  lifeCurveInput: string
  lifetimeModel: string
  modelParamsInput: Record<string, string>
  safetyFactor: string

  /* Results */
  tjSeries: number[] | null
  result: RainflowResult | null
  allTjSeries: Record<string, number[]> | null
}

export interface RainflowPageActions {
  /** Patch one or more fields (shallow merge). */
  patch: (partial: Partial<RainflowPageState>) => void
  /** Reset everything to initial defaults. */
  reset: () => void
}

/* ---------- Defaults ---------- */
const defaultZthSelf = '0.05,0.001\n0.15,0.01\n0.30,0.1\n0.50,1.0'
const defaultZthCross = '0.01,0.005\n0.03,0.05'

const initialState: RainflowPageState = {
  tab: 0,
  loading: false,
  error: null,

  inputMode: 'power',
  zthMode: 'foster',
  fosterInput: defaultZthSelf,
  zthInput: '',
  dt: '1.0',
  ambientTemp: '25',

  sourceCount: 1,
  sourceNames: ['IGBT', 'Diode'],
  sourcePowerData: ['', ''],
  zthMatrixInput: [
    [defaultZthSelf, defaultZthCross],
    [defaultZthCross, defaultZthSelf],
  ],
  targetNode: 0,

  nBand: '20',
  yMin: '',
  yMax: '',
  ignoreBelow: '0',
  rearrange: false,

  damageMethod: 'life_curve',
  lifeCurveInput: '20,500000\n40,120000\n80,30000',
  lifetimeModel: 'coffin-manson',
  modelParamsInput: { A: '3.025e14', alpha: '5.039' },
  safetyFactor: '1.0',

  tjSeries: null,
  result: null,
  allTjSeries: null,
}

/* ---------- Store ---------- */
export const useRainflowStore = create<RainflowPageState & RainflowPageActions>()(
  (set) => ({
    ...initialState,
    patch: (partial) => set(partial),
    reset: () => set(initialState),
  }),
)

export default useRainflowStore
