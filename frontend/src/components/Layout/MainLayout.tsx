import React, { useState } from 'react'
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Timeline as PredictionIcon,
  Waves as RainflowIcon,
  ShowChart as DamageIcon,
  ChevronLeft as ChevronLeftIcon,
  Science as FittingIcon,
} from '@mui/icons-material'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'

const DRAWER_WIDTH = 260

interface MenuItemType {
  id: string
  label: string
  icon: React.ElementType
  path: string
}

const menuItems: MenuItemType[] = [
  { id: 'home', label: '首页', icon: HomeIcon, path: '/' },
  { id: 'fitting', label: '参数拟合', icon: FittingIcon, path: '/fitting' },
  { id: 'prediction', label: '寿命预测', icon: PredictionIcon, path: '/prediction' },
  { id: 'rainflow', label: '雨流计数', icon: RainflowIcon, path: '/rainflow' },
  { id: 'damage', label: '累计损伤与寿命', icon: DamageIcon, path: '/damage' },
]

export const MainLayout: React.FC = () => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const [mobileOpen, setMobileOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(!isMobile)
  const navigate = useNavigate()
  const location = useLocation()

  const handleDrawerToggle = () => {
    if (isMobile) {
      setMobileOpen(!mobileOpen)
    } else {
      setDrawerOpen(!drawerOpen)
    }
  }

  const handleMenuItemClick = (path: string) => {
    navigate(path)
    if (isMobile) {
      setMobileOpen(false)
    }
  }

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
        }}
      >
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600 }}>
          CIPS 2008
        </Typography>
        {!isMobile && (
          <IconButton onClick={handleDrawerToggle} size="small">
            <ChevronLeftIcon />
          </IconButton>
        )}
      </Toolbar>
      <List sx={{ flex: 1, px: 1 }}>
        {menuItems.map((item) => {
          const Icon = item.icon
          const isSelected = location.pathname === item.path
          return (
            <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => handleMenuItemClick(item.path)}
                selected={isSelected}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '& .MuiSvgIcon-root': {
                      color: 'primary.contrastText',
                    },
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 40,
                    color: isSelected ? 'inherit' : 'text.secondary',
                  }}
                >
                  <Icon />
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  sx={{
                    '& .MuiTypography-root': {
                      fontWeight: isSelected ? 600 : 400,
                    },
                  }}
                />
              </ListItemButton>
            </ListItem>
          )
        })}
      </List>
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary">
          Version 1.0.0
        </Typography>
      </Box>
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerOpen ? DRAWER_WIDTH : 0}px)` },
          ml: { md: drawerOpen ? `${DRAWER_WIDTH}px` : 0 },
          transition: 'width 0.3s, margin-left 0.3s',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            CIPS 2008 寿命预测软件
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Mobile drawer */}
      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="persistent"
          open={drawerOpen}
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerOpen ? DRAWER_WIDTH : 0}px)` },
          transition: 'width 0.3s',
          height: '100vh',
          overflow: 'auto',
          backgroundColor: 'background.default',
        }}
      >
        <Toolbar />
        <Box sx={{ p: 3 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
}

export default MainLayout
