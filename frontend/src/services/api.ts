import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ApiResponse } from '@/types'

/**
 * API Service for backend communication
 * Handles all HTTP requests to the CIPS 2008 backend
 */

/**
 * Custom API Error class for better error handling
 */
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public originalError?: unknown
  ) {
    super(message)
    this.name = 'APIError'
  }
}

/**
 * Safe localStorage wrapper with error handling
 */
class StorageService {
  getItem(key: string): string | null {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return null
      }
      return localStorage.getItem(key)
    } catch (error) {
      console.warn(`Failed to read localStorage key "${key}":`, error)
      return null
    }
  }

  setItem(key: string, value: string): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return false
      }
      localStorage.setItem(key, value)
      return true
    } catch (error) {
      console.warn(`Failed to write localStorage key "${key}":`, error)
      return false
    }
  }

  removeItem(key: string): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return false
      }
      localStorage.removeItem(key)
      return true
    } catch (error) {
      console.warn(`Failed to remove localStorage key "${key}":`, error)
      return false
    }
  }
}

const storageService = new StorageService()

class ApiService {
  private client: AxiosInstance
  private controller: AbortController | null = null

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = storageService.getItem('auth_token')
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Handle common errors
        if (error.response) {
          switch (error.response.status) {
            case 401:
              // Unauthorized - clear token and redirect to login
              storageService.removeItem('auth_token')
              window.location.href = '/login'
              break
            case 403:
              console.error('Access forbidden')
              break
            case 404:
              console.error('Resource not found')
              break
            case 500:
              console.error('Server error')
              break
          }
        }
        // Wrap error in APIError for consistent handling
        const apiError = new APIError(
          error.response?.data?.message || error.message || 'An error occurred',
          error.response?.status,
          error
        )
        return Promise.reject(apiError)
      }
    )
  }

  /**
   * Cancel all pending requests
   */
  cancelPendingRequests(): void {
    if (this.controller) {
      this.controller.abort()
    }
    this.controller = new AbortController()
  }

  /**
   * Get the current abort signal for cancellation
   */
  private getAbortSignal(): AbortSignal | undefined {
    if (!this.controller) {
      this.controller = new AbortController()
    }
    return this.controller.signal
  }

  /**
   * Make a GET request
   */
  private async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<ApiResponse<T>> = await this.client.get(url, {
        ...config,
        signal: this.getAbortSignal(),
      })
      return response.data
    } catch (error) {
      if (axios.isCancel(error)) {
        throw new APIError('Request was cancelled')
      }
      throw error
    }
  }

  /**
   * Make a POST request
   */
  private async post<T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<ApiResponse<T>> = await this.client.post(url, data, {
        ...config,
        signal: this.getAbortSignal(),
      })
      return response.data
    } catch (error) {
      if (axios.isCancel(error)) {
        throw new APIError('Request was cancelled')
      }
      throw error
    }
  }

  /**
   * Make a PUT request
   */
  private async put<T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<ApiResponse<T>> = await this.client.put(url, data, {
        ...config,
        signal: this.getAbortSignal(),
      })
      return response.data
    } catch (error) {
      if (axios.isCancel(error)) {
        throw new APIError('Request was cancelled')
      }
      throw error
    }
  }

  /**
   * Make a DELETE request
   */
  private async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<ApiResponse<T>> = await this.client.delete(url, {
        ...config,
        signal: this.getAbortSignal(),
      })
      return response.data
    } catch (error) {
      if (axios.isCancel(error)) {
        throw new APIError('Request was cancelled')
      }
      throw error
    }
  }

  // ============================================
  // Lifetime Prediction Endpoints
  // ============================================

  /**
   * Perform lifetime prediction using specified model
   */
  async predictLifetime(request: {
    modelType: string
    params: Record<string, unknown>
    cycles: Array<{ Tmax: number; Tmin: number; theating: number }>
  }) {
    return this.post('/prediction/predict', request)
  }

  /**
   * Get available lifetime models
   */
  async getModels() {
    return this.get('/prediction/models')
  }

  /**
   * Get default parameters for a model
   */
  async getModelDefaultParams(modelType: string) {
    return this.get(`/prediction/models/${modelType}/defaults`)
  }

  // ============================================
  // Rainflow Counting Endpoints
  // ============================================

  /**
   * Perform rainflow cycle counting
   */
  async performRainflowCounting(data: {
    timeSeries: number[]
    binCount?: number
  }) {
    const payload = {
      data_points: data.timeSeries.map((value, index) => ({ time: index, value })),
      bin_count: data.binCount ?? 20,
      method: 'ASTM',
    }

    try {
      const response = await this.client.post('/rainflow/count', payload, {
        signal: this.getAbortSignal(),
      })

      const backendData = response.data as {
        cycles?: Array<{ stress_range: number; mean_value: number; cycles: number }>
        total_cycles?: number
        max_range?: number
        summary?: Record<string, unknown>
      }

      const cycles = (backendData.cycles ?? []).map((cycle) => ({
        range: cycle.stress_range,
        mean: cycle.mean_value,
        count: cycle.cycles,
        cycleType: cycle.cycles >= 1 ? 'full' : 'half',
      }))

      const result = {
        originalData: data.timeSeries,
        cycles,
        totalCycles: backendData.total_cycles ?? 0,
        maxRange: backendData.max_range ?? 0,
        minRange: cycles.length > 0 ? Math.min(...cycles.map((c) => c.range)) : 0,
        binCount: data.binCount ?? 20,
        summary: backendData.summary ?? {},
      }

      return {
        success: true,
        data: result,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : '雨流计数请求失败'
      return {
        success: false,
        error: message,
      }
    }
  }

  /**
   * Run one-stop rainflow pipeline
   */
  async runRainflowPipeline(data: {
    junctionTemperature?: number[]
    powerCurve?: number[]
    thermalImpedanceCurve?: number[]
    fosterParams?: Array<{ R: number; tau: number }>
    // Multi-source
    powerCurves?: number[][]
    zthMatrix?: Array<Array<Array<{ R: number; tau: number }>>>
    sourceNames?: string[]
    targetNode?: number
    // Common
    ambientTemperature?: number
    responseType?: 'impulse' | 'step'
    dt?: number
    binCount?: number
    rearrange?: boolean
    nBand?: number
    yMin?: number
    yMax?: number
    ignoreBelow?: number
    // Damage: lifetime model
    lifetimeModel?: string
    modelParams?: Record<string, number>
    safetyFactor?: number
  }) {
    const payload: Record<string, unknown> = {
      junction_temperature: data.junctionTemperature,
      power_curve: data.powerCurve,
      thermal_impedance_curve: data.thermalImpedanceCurve,
      foster_params: data.fosterParams?.map((e) => ({ R: e.R, tau: e.tau })),
      // multi-source
      power_curves: data.powerCurves,
      zth_matrix: data.zthMatrix?.map((row) =>
        row.map((cell) => cell.map((e) => ({ R: e.R, tau: e.tau }))),
      ),
      source_names: data.sourceNames,
      target_node: data.targetNode,
      // common
      ambient_temperature: data.ambientTemperature ?? 25,
      response_type: data.responseType ?? 'impulse',
      dt: data.dt ?? 1.0,
      bin_count: data.binCount ?? 20,
      rearrange: data.rearrange ?? false,
      n_band: data.nBand ?? 20,
      y_min: data.yMin,
      y_max: data.yMax,
      ignore_below: data.ignoreBelow ?? 0,
      // damage: model
      lifetime_model: data.lifetimeModel,
      model_params: data.modelParams,
      safety_factor: data.safetyFactor,
    }

    // Remove undefined keys to keep payload clean
    Object.keys(payload).forEach((k) => {
      if (payload[k] === undefined) delete payload[k]
    })

    try {
      const response = await this.client.post('/rainflow/pipeline', payload, {
        signal: this.getAbortSignal(),
      })

      const backendData = response.data as {
        junction_temperature: number[]
        thermal_summary?: Record<string, number>
        cycles?: Array<{ stress_range: number; mean_value: number; cycles: number }>
        matrix_rows?: Array<{ delta_tj: number; mean_tj: number; count: number }>
        total_cycles?: number
        max_range?: number
        summary?: Record<string, unknown>
        damage?: Record<string, unknown> | null
        from_to_matrix?: Record<string, unknown> | null
        amplitude_histogram?: Record<string, unknown> | null
        residual?: number[] | null
        all_junction_temperatures?: Record<string, number[]> | null
        model_damage?: Record<string, unknown> | null
      }

      const cycles = (backendData.cycles ?? []).map((cycle) => ({
        range: cycle.stress_range,
        mean: cycle.mean_value,
        count: cycle.cycles,
        cycleType: cycle.cycles >= 1 ? 'full' : 'half',
      }))

      const result = {
        originalData: backendData.junction_temperature ?? data.junctionTemperature ?? [],
        cycles,
        totalCycles: backendData.total_cycles ?? 0,
        maxRange: backendData.max_range ?? 0,
        minRange: cycles.length > 0 ? Math.min(...cycles.map((c) => c.range)) : 0,
        binCount: data.binCount ?? 20,
        summary: backendData.summary ?? {},
        damage: backendData.damage ?? null,
        thermalSummary: backendData.thermal_summary ?? null,
        fromToMatrix: (backendData.from_to_matrix as any) ?? null,
        amplitudeHistogram: (backendData.amplitude_histogram as any) ?? null,
        residual: backendData.residual ?? null,
        allJunctionTemperatures: backendData.all_junction_temperatures ?? null,
        modelDamage: (backendData.model_damage as any) ?? null,
      }

      return { success: true, data: result }
    } catch (error) {
      const message = error instanceof Error ? error.message : '雨流pipeline请求失败'
      return { success: false, error: message }
    }
  }

  /**
   * Get rainflow result by ID
   */
  async getRainflowResult(id: string) {
    return this.get(`/rainflow/results/${id}`)
  }

  // ============================================
  // Damage Accumulation Endpoints
  // ============================================

  /**
   * Calculate cumulative damage using Miner's rule
   */
  async calculateDamage(request: {
    modelType: string
    params: Record<string, unknown>
    cycles: Array<{
      Tmax: number
      Tmin: number
      theating: number
      count: number
    }>
  }) {
    return this.post('/damage/calculate', request)
  }

  /**
   * Get remaining life estimation
   */
  async getRemainingLife(damageResultId: string) {
    return this.get(`/damage/remaining/${damageResultId}`)
  }

  // ============================================
  // Analysis Endpoints
  // ============================================

  /**
   * Perform sensitivity analysis
   */
  async performSensitivityAnalysis(request: {
    modelType: string
    baseParams: Record<string, unknown>
    parametersToAnalyze: Array<{
      name: string
      variation: { min: number; max: number }
      steps: number
    }>
  }) {
    return this.post('/analysis/sensitivity', request)
  }

  /**
   * Perform Weibull analysis
   */
  async performWeibullAnalysis(data: {
    failures: number[]
    censored?: number[]
    confidenceLevel?: number
  }) {
    return this.post('/analysis/weibull', data)
  }

  // ============================================
  // Data Management Endpoints
  // ============================================

  /**
   * Upload experiment data file
   */
  async uploadExperimentData(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    return this.post('/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  }

  /**
   * Get all experiment data
   */
  async getExperimentData(page = 1, limit = 20) {
    return this.get('/data/experiments', { params: { page, limit } })
  }

  /**
   * Delete experiment data
   */
  async deleteExperimentData(id: string) {
    return this.delete(`/data/experiments/${id}`)
  }

  /**
   * Export data in specified format
   */
  async exportData(request: {
    type: 'prediction' | 'rainflow' | 'damage' | 'analysis'
    id: string
    format: 'csv' | 'xlsx' | 'json' | 'pdf'
    includeCharts?: boolean
  }) {
    return this.post('/data/export', request, {
      responseType: 'blob',
    })
  }

  /**
   * Export prediction as PDF report
   */
  async exportPredictionPDF(data: {
    prediction: {
      id?: number
      name: string
      model_type: string
      predicted_lifetime_years?: number
      predicted_lifetime_cycles?: number
      total_damage?: number
      confidence_level?: number
      created_at?: string
    }
    parameters: Record<string, unknown>
    mission_profile?: Record<string, unknown>
    config?: {
      include_charts?: boolean
      include_confidence?: boolean
      language?: string
      page_size?: string
    }
  }) {
    return this.post('/export/report/pdf', data, {
      responseType: 'blob',
      headers: { 'Content-Type': 'application/json' },
    })
  }

  /**
   * Export prediction as Excel report
   */
  async exportPredictionExcel(data: {
    prediction: {
      id?: number
      name: string
      model_type: string
      predicted_lifetime_years?: number
      predicted_lifetime_cycles?: number
      total_damage?: number
      confidence_level?: number
      created_at?: string
    }
    parameters: Record<string, unknown>
    mission_profile?: Record<string, unknown>
    rainflow_cycles?: Array<{
      stress_range: number
      mean_value: number
      cycles: number
      damage?: number
    }>
    include_rainflow?: boolean
    include_mission_profile?: boolean
  }) {
    return this.post('/export/report/excel', data, {
      responseType: 'blob',
      headers: { 'Content-Type': 'application/json' },
    })
  }

  /**
   * Export prediction by ID as PDF
   */
  async exportPredictionPDFById(id: number) {
    return this.client.get(`/export/export/prediction/${id}/pdf`, {
      responseType: 'blob',
    })
  }

  /**
   * Export prediction by ID as Excel
   */
  async exportPredictionExcelById(id: number) {
    return this.client.get(`/export/export/prediction/${id}/excel`, {
      responseType: 'blob',
    })
  }

  /**
   * Export experiment by ID as PDF
   */
  async exportExperimentPDFById(id: number) {
    return this.client.get(`/export/export/experiment/${id}/pdf`, {
      responseType: 'blob',
    })
  }

  /**
   * Download file from blob response
   */
  downloadBlob(blob: Blob, filename: string) {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  }

  /**
   * Get prediction history
   */
  async getPredictionHistory(page = 1, limit = 20) {
    return this.get('/data/predictions', { params: { page, limit } })
  }

  // ============================================
  // Settings Endpoints
  // ============================================

  /**
   * Get application settings
   */
  async getSettings() {
    return this.get('/settings')
  }

  /**
   * Update application settings
   */
  async updateSettings(settings: Record<string, unknown>) {
    return this.put('/settings', settings)
  }

  /**
   * Reset settings to defaults
   */
  async resetSettings() {
    return this.post('/settings/reset', {})
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
