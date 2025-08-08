"""
This module contains functions to generate traces, prefixes, and next activity 
as tensors.

Functions:
    create_trace
    create_prefix
    create_next_act   
"""

import pandas as pd
import numpy as np
import torch

def create_trace(df,
                 trace_len,
                 case_list,
                 start_idx,
                 end_idx,
                 trace_col_name,
                 categorical_features,
                 case_id,
                 event_idx,
                 pad_position='right'):
    """
    Create trace tensor.

    Parameters
    ----------
    df : pandas DataFrame
        Event log.
    trace_len : int
        The maximum length of the trace plus 1.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace prefix.
    start_idx : int
        Index of the start of the range (inclusive).
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Prefixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For training/validation set, this should be train_test_split_idx.
    trace_col_name : list
        Name(s) of column(s) containing features used in the trace prefix.
    categorical_features : list
        Name(s) of column(s) containing categorical features.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.
    pad_position : str
        'right' or 'left'

    Returns
    -------
    tensors_list : list of torch.Tensor
        Each tensor is of shape (num_obs, trace_len). One tensor is returned per 
        feature column.
    """

    df = df.sort_values(by=[case_id, event_idx]).reset_index(drop=True)
    
    tensors_list = []

    for col in trace_col_name:

        trace_list = []

        # set padding values for categorical and continuous features
        padding_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx-1):

            if df[case_id].iloc[i] in case_list and df[case_id].iloc[i] != df[case_id].iloc[i + 1]:
 
                current_event_idx = df[event_idx].iloc[i]
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)                
                trace = filtered_df[filtered_df[event_idx] <= current_event_idx][col].tolist()

                if len(trace) <= 2:
                    continue

                trace = trace[1:] # remove SOC

                assert len(trace) <= trace_len, "len(trace) should not exceed trace_len"
                
                padding = [padding_number] * max(0, trace_len - len(trace))
                if pad_position == 'right':
                    trace = trace + padding
                elif pad_position == 'left':
                    trace = padding + trace
                
                trace_list.append(trace)

        # create tensor for each feature column
        trace_tensor = torch.tensor(trace_list)

        if col in categorical_features:
            trace_tensor = trace_tensor.long() # shape: (num_obs, trace_len)
        else:
            trace_tensor = trace_tensor.float() # shape: (num_obs, trace_len)

        tensors_list.append(trace_tensor)

    return tensors_list

def create_prefix(df,
                  trace_prefix_len, 
                  case_list,
                  start_idx,
                  end_idx,
                  trace_col_name,
                  categorical_features,
                  case_id,
                  event_name,
                  event_idx,
                  pad_position):
    """
    Create prefix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log.
    trace_prefix_len : int
        The maximum length of the trace prefix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace prefix.
    start_idx : int
        Index of the start of the range (inclusive).
        For the test set, this should be train_test_split_idx.
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Prefixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For the test set, this should be the index of the first event after 
        end_timestamp.
        For training/validation set, this should be train_test_split_idx.
    trace_col_name : list
        Name(s) of column(s) containing features used in the trace prefix.
    categorical_features : list
        Name(s) of column(s) containing categorical features.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.
    pad_position : str
        'right' or 'left'

    Returns
    -------
    tensors_list : list of torch.Tensor
        Each tensor is of shape (num_obs, prefix_len). One tensor is returned per 
        feature column.
    """

    # ensure event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)
    
    tensors_list = []

    for col in trace_col_name:

        trace_prefix_list = []

        # set padding values for categorical and continuous features
        padding_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):

            # skip SOC (event_name == 2) and EOC events (event_name == 3)
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] not in {2, 3}:
 
                current_event_idx = df[event_idx].iloc[i]

                # get all events of the current case
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                
                # get events up to and including the current event
                prefix = filtered_df[filtered_df[event_idx] <= current_event_idx][col].tolist() 
                
                prefix = prefix[1:] # remove SOC

                assert len(prefix) <= trace_prefix_len, "len(prefix) should not exceed trace_prefix_len"

                padding = [padding_number] * max(0, trace_prefix_len - len(prefix)) 
                if pad_position == 'right':
                    prefix =  prefix + padding
                elif pad_position == 'left':
                    prefix = padding + prefix
                
                trace_prefix_list.append(prefix)

        # create tensor for each feature column
        trace_prefix_tensor = torch.tensor(trace_prefix_list)

        if col in categorical_features:
            trace_prefix_tensor = trace_prefix_tensor.long() # shape: (num_obs, prefix_len)
        else:
            trace_prefix_tensor = trace_prefix_tensor.float() # shape: (num_obs, prefix_len)

        tensors_list.append(trace_prefix_tensor)

    return tensors_list

def create_next_act(df, 
                    case_list, 
                    start_idx,
                    end_idx,
                    case_id, 
                    event_name,
                    event_idx):
    """
    Create next activity tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace suffix.
    start_idx : int
        Index of the start of the range (inclusive).
        For the test set, this should be train_test_split_idx.
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Suffixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For the test set, this should be the index of the first event after 
        end_timestamp.
        For training/validation set, this should be train_test_split_idx.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.

    Returns
    -------
    next_act_tensor : torch.Tensor
        A tensor storing next activities.
        Shape: (num_samples, )
    """

    # ensure event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    next_act_list = []

    for i in range(start_idx, end_idx):
        # skip SOC (event_name == 2) and EOC events (event_name == 3)
        if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] not in {2, 3}:

            current_event_idx = df[event_idx].iloc[i]
            # get all events of the current case
            filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)

            next_events = filtered_df[filtered_df[event_idx] > current_event_idx]

            if not next_events.empty:
                next_act = next_events.iloc[0][event_name]
                next_act_list.append(next_act)

            else:
                pass

    # create tensor for each feature column    
    next_act_tensor = torch.tensor(next_act_list)

    return next_act_tensor