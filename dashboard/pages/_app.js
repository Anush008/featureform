import Container from '@mui/material/Container';
import CssBaseline from '@mui/material/CssBaseline';
import { makeStyles, ThemeProvider } from '@mui/styles';
import React from 'react';
import ResourcesAPI from '../src/api/resources';
import ReduxStore from '../src/components/redux/store';
import ReduxWrapper from '../src/components/redux/wrapper';
import SideNav from '../src/components/sideNav/SideNav';
import TopBar from '../src/components/topbar/TopBar';
import '../src/styles/base.css';
import theme from '../src/styles/theme';

const apiHandle = new ResourcesAPI();
const useStyles = makeStyles(() => ({
  pageContainer: {
    height: '100%',
    width: '100%',
    top: '70px',
    position: 'relative',
  },
}));

export const MyApp = ({ Component, pageProps }) => {
  const classes = useStyles();
  return (
    <React.StrictMode>
      <ReduxWrapper store={ReduxStore}>
        <ThemeWrapper>
          <TopBar className={classes.topbar} api={apiHandle} />
          <SideNav>
            <Container
              maxWidth='xl'
              className={classes.root}
              classes={{ maxWidthXl: classes.pageContainer }}
            >
              <Component {...pageProps} api={apiHandle} />
            </Container>
          </SideNav>
        </ThemeWrapper>
      </ReduxWrapper>
    </React.StrictMode>
  );
};

export const views = {
  RESOURCE_LIST: 'ResourceList',
  EMPTY: 'Empty',
};

export const ThemeWrapper = ({ children }) => (
  <ThemeProvider theme={theme}>
    <CssBaseline />
    {children}
  </ThemeProvider>
);

export default MyApp;

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
// serviceWorker.unregister();
