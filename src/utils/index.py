from seiz_eeg.schemas import ClipsDF
import pandas as pd
from typing import Optional

def ensure_eeg_multiindex(df: pd.DataFrame, id_col_name: Optional[str] = 'id') -> pd.DataFrame:
    """
    Ensures the DataFrame has a MultiIndex with levels ['patient', 'session', 'clip', 'segment'].
    ID format expected: <patient_id>_s<session_no>_t<clip_no>_<segment_no>
    Uses ClipsDF constants for index level names.
    """
    df_out = df.copy()
    desired_names = [ClipsDF.patient, ClipsDF.session, ClipsDF.segment]

    if isinstance(df_out.index, pd.MultiIndex) and list(df_out.index.names) == desired_names:
        return df_out

    id_series: Optional[pd.Series] = None
    if id_col_name and id_col_name in df_out.columns:
        id_series = df_out[id_col_name]
    elif not isinstance(df_out.index, pd.MultiIndex) and df_out.index.dtype == 'object': # Check if index is string-like
        id_series = pd.Series(df_out.index.values, index=df_out.index)
    else: # Try to reset index if it's a simple RangeIndex or similar, to find 'id' column
        df_temp = df_out.reset_index()
        if id_col_name and id_col_name in df_temp.columns:
            id_series = df_temp[id_col_name]
        else: # Fallback if 'id' column name is not standard or index is not string based.
            raise ValueError("Cannot determine ID source for MultiIndex creation. "
                             "Provide string IDs in a column (specify with 'id_col_name') or as a simple string-based index.")


    if id_series is None or id_series.empty:
        raise ValueError("Could not determine a valid source for string IDs for MultiIndex creation.")

    parsed_ids = []
    for record_id_val in id_series:
        record_id_str = str(record_id_val)
        parts = record_id_str.split('_')
        if len(parts) != 3: # e.g. P001_s1_t1_0
            # Try common alternative P001_s01_clip1_seg0
            parts_alt_clip = record_id_str.replace('_clip', '_c').replace('_seg','_s').split('_')
            if len(parts_alt_clip) == 4 and parts_alt_clip[2].startswith('c') and parts_alt_clip[3].startswith('s'):
                parts = parts_alt_clip
            else:
                 raise ValueError(f"ID '{record_id_str}' malformed. Expected 4 parts (e.g., pat_sS_tT_Seg or pat_sS_cC_sS), got {len(parts)}.")
        try:
            patient = parts[0]
            session = int(parts[1][1:]) # Remove 's'
            clip_val_str = parts[2]
            clip_val = int(clip_val_str[1:]) if clip_val_str.startswith(('t', 'c')) else int(clip_val_str) # Remove 't' or 'c'
            
            segment_val_str = parts[3]
            segment_val = int(segment_val_str[1:]) if segment_val_str.startswith('s') else int(segment_val_str) # Remove 's' if present
            parsed_ids.append((patient, session, clip_val, segment_val))
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing components of ID '{record_id_str}': {e}") from e
    
    if not parsed_ids: # Should not happen if id_series was not empty
        df