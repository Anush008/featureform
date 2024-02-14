import DoubleArrowIcon from '@mui/icons-material/DoubleArrow';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  Box,
  CircularProgress,
  Grid,
  IconButton,
  TextField,
  Typography,
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import { useDataAPI } from 'hooks/dataAPI';
import React, { useEffect, useState } from 'react';
import { useStyles } from './styles';

export default function TaskRunCard({ handleClose, searchId }) {
  const classes = useStyles();
  const dataAPI = useDataAPI();
  const [taskRecord, setTaskRecord] = useState({});
  const [loading, setLoading] = useState(true);

  const columns = [
    {
      field: 'id',
      headerName: 'runId',
      width: 1,
      editable: false,
      sortable: false,
      filterable: false,
      hide: true,
      hideable: false,
      display: false,
    },
    {
      field: 'lastRunTime',
      headerName: 'Last Run',
      sortable: false,
      filterable: false,
      width: 200,
      valueGetter: (params) => {
        return new Date(params?.row?.lastRunTime)?.toLocaleString();
      },
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 200,
      editable: false,
      sortable: false,
      filterable: false,
    },
    {
      field: 'link',
      headerName: 'Link',
      width: 250,
      editable: false,
      sortable: false,
      filterable: false,
    },
  ];

  useEffect(async () => {
    if (searchId && loading) {
      let data = await dataAPI.getTaskRunDetails(searchId);
      setTaskRecord(data);
      setTimeout(() => {
        setLoading(false);
      }, 750);
    }
  }, [searchId, loading]);

  const handleReloadRequest = () => {
    if (!loading) {
      setLoading(true);
    }
  };

  return (
    <Box className={classes.taskCardBox}>
      <Box style={{ float: 'left' }}>
        <IconButton variant='' size='large' onClick={() => handleClose()}>
          <DoubleArrowIcon />
        </IconButton>
      </Box>
      <Box style={{ float: 'right' }}>
        <IconButton variant='' size='large' onClick={handleReloadRequest}>
          {loading ? <CircularProgress size={'.75em'} /> : <RefreshIcon />}
        </IconButton>
      </Box>
      <Grid style={{ padding: 12 }} container>
        <Grid item xs={6} justifyContent='flex-start'>
          <Typography variant='h5'>{taskRecord.name}</Typography>
        </Grid>
        <Grid item xs={6} justifyContent='center'>
          <Typography variant='h6'>Status: {taskRecord.status}</Typography>
        </Grid>
        <Grid
          item
          xs={12}
          justifyContent='flex-start'
          style={{ paddingTop: 20 }}
        >
          <Typography variant='h6'>Logs/Errors</Typography>
        </Grid>
        <Grid item xs={12} justifyContent='flex-start'>
          <Typography>
            <TextField
              style={{ width: '100%' }}
              variant='filled'
              disabled
              value={taskRecord.logs}
              multiline
              minRows={3}
            ></TextField>
          </Typography>
        </Grid>
        <Grid item xs={12} justifyContent='center' style={{ paddingTop: 20 }}>
          <Typography variant='h6'>Task Run Details</Typography>
        </Grid>
        <Grid item xs={12} justifyContent='center'>
          <Typography>
            <TextField
              style={{ width: '100%' }}
              variant='filled'
              disabled
              value={taskRecord.details}
              multiline
              minRows={3}
            ></TextField>
          </Typography>
        </Grid>
        <Grid
          item
          xs={12}
          justifyContent='flex-start'
          style={{ paddingTop: 20 }}
        >
          <Typography variant='h6'>Other Runs</Typography>
        </Grid>
        <Grid item xs={12} justifyContent='flex-start'></Grid>
        <Grid item xs={12} justifyContent='center'>
          <DataGrid
            density='compact'
            autoHeight
            aria-label='Other Runs'
            rows={taskRecord?.otherRuns ?? []}
            rowsPerPageOptions={[5]}
            columns={columns}
            initialState={{
              pagination: { paginationModel: { page: 0, pageSize: 5 } },
            }}
            pageSize={5}
          />
        </Grid>
      </Grid>
    </Box>
  );
}