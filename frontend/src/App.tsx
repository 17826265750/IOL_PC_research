import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, CssBaseline, Box } from '@mui/material'

import MainLayout from '@/components/Layout/MainLayout'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import {
  Home,
  Prediction,
  Analysis,
  RainflowCounting,
  DamageAccumulation,
  RemainingLife,
  ParameterFitting
} from '@/pages'
import theme from '@/theme/theme'

// Placeholder pages - these will be implemented in subsequent tasks
const PlaceholderPage: React.FC<{ title: string }> = ({ title }) => (
  <Box sx={{ p: 3 }}>
    <h1>{title}</h1>
    <p>This page is under development.</p>
  </Box>
)

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<MainLayout />}>
              <Route index element={<Home />} />
              <Route path="prediction" element={<Prediction />} />
              <Route path="fitting" element={<ParameterFitting />} />
              <Route path="rainflow" element={<RainflowCounting />} />
              <Route path="damage" element={<DamageAccumulation />} />
              <Route path="remaining" element={<RemainingLife />} />
              <Route path="analysis" element={<Analysis />} />
              <Route path="data" element={<PlaceholderPage title="数据管理" />} />
              <Route path="settings" element={<PlaceholderPage title="设置" />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </ErrorBoundary>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
