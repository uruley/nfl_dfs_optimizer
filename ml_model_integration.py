import pandas as pd
from Scripts.live_qb_features import create_live_qb_features

def integrate_ml_with_ui(players_df):
    """
    Integrates ML projections from create_live_qb_features() into the DraftKings players dataframe.
    
    Args:
        players_df (pd.DataFrame): DraftKings CSV dataframe with 'Name' and 'Projection' columns
        
    Returns:
        pd.DataFrame: Updated dataframe with ML projections in the 'Projection' column
    """
    # Get ML projections and features
    features_df, projections_array = create_live_qb_features()
    
    # Create a list of player names from the known output
    player_names = ["Jalen Hurts", "Dak Prescott", "Tanner McKee", "Joe Milton III", 
                    "Dorian Thompson-Robinson", "Kyle McCord", "Will Grier"]
    
    # Create a dictionary mapping player names to projections
    projection_dict = dict(zip(player_names, projections_array))
    
    # Create a copy of input dataframe to avoid modifying the original
    result_df = players_df.copy()
    
    # Standardize names for matching: lowercase, remove extra spaces
    def standardize_name(name):
        return ' '.join(str(name).strip().lower().split())
    
    # Create a column for standardized names
    result_df['standardized_name'] = result_df['Name'].apply(standardize_name)
    
    # Create a mapping function that handles partial matches
    def map_projection(player_name):
        standardized = standardize_name(player_name)
        # Exact match
        for ml_name, projection in projection_dict.items():
            if standardize_name(ml_name) == standardized:
                return projection
            # Partial match (last name or full name parts)
            if ml_name.split()[-1].lower() in standardized:
                return projection
        # Return original projection if no match found
        return result_df.loc[result_df['standardized_name'] == standardized, 'Projection'].iloc[0] if not result_df.loc[result_df['standardized_name'] == standardized, 'Projection'].empty else 0
    
    # Apply projections
    result_df['Projection'] = result_df['Name'].apply(map_projection)
    
    # Drop temporary column
    result_df = result_df.drop(columns=['standardized_name'])
    
    return result_df