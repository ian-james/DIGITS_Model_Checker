from collections import OrderedDict
from datetime import datetime, timezone

import pandas as pd

def setup_frame_data(frame_idx):
    od = OrderedDict()
    od['frame_idx'] = frame_idx
    od['timestamp'] = datetime.now(timezone.utc).isoformat()

def add_dataframe(df, frame_data):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame([frame_data], columns=frame_data.keys())
    else:
        sfd = pd.DataFrame([frame_data], columns=frame_data.keys())
        df = pd.concat( [df,sfd], ignore_index=True,)
    return df


def get_key_frames(df):
    if df is not None and not df.empty:
        cols = df.columns
        key_columns = ['fps_count', 'timestamp', 'flexion_left',
                    'flexion_right', 'abduction_left', 'abduction_right']
        # Select only the key_columns from the data frame if they exist.
        df = df.loc[:, df.columns.isin(key_columns)]
    return df
