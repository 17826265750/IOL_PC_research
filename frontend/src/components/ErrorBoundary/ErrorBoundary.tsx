import React from 'react'
import { Box, Typography, Button, Paper } from '@mui/material'

interface Props {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }
      return (
        <Box sx={{ p: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <Paper sx={{ p: 4, maxWidth: 500, textAlign: 'center' }}>
            <Typography variant="h5" color="error" gutterBottom>
              出现错误 / Error Occurred
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </Typography>
            <Button variant="contained" onClick={this.handleReset}>
              返回首页 / Return Home
            </Button>
          </Paper>
        </Box>
      )
    }
    return this.props.children
  }
}

export default ErrorBoundary
