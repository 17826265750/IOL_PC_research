/**
 * 功率模块寿命分析软件 - 应用入口
 * @author GSH
 */
import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, CssBaseline } from '@mui/material'

import MainLayout from '@/components/Layout/MainLayout'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import {
  Home,
  ParameterFitting,
  Prediction,
  RainflowCounting,
  DamageAssessment,
} from '@/pages'
import theme from '@/theme/theme'

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<MainLayout />}>
              <Route index element={<Home />} />
              <Route path="fitting" element={<ParameterFitting />} />
              <Route path="prediction" element={<Prediction />} />
              <Route path="rainflow" element={<RainflowCounting />} />
              <Route path="damage" element={<DamageAssessment />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </ErrorBoundary>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
