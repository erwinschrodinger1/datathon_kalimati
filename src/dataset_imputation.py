import pandas as pd


def dataset_imputation(df):
    # Group the dataframe by commodity
    grouped_df = df.groupby("Commodity")

    # Create a list to collect all new rows
    new_rows = []

    # Iterate over each group
    for commodity, group in grouped_df:
        # Get the full date range for the group
        full_date_range = pd.date_range(
            start=group["Date"].min(), end=group["Date"].max(), freq="D"
        )

        # Find the missing dates
        missing_dates = full_date_range.difference(group["Date"].unique())

        # Create new rows for the missing dates
        for date in missing_dates:
            new_row = {"Commodity": commodity, "Date": date}

            # Set the other columns to 0 or any desired fill value
            for column in df.columns:
                if column not in ["Commodity", "Date"]:
                    new_row[column] = 0

            # Add the new row to the list of new rows
            new_rows.append(new_row)

    # Create a new dataframe from the new rows
    new_rows_df = pd.DataFrame(new_rows)

    # Concatenate the new rows with the original dataframe
    df_filled = pd.concat([df, new_rows_df], ignore_index=True)

    # Sort the dataframe by 'Commodity' and 'Date'
    df_filled.sort_values(by=["Commodity", "Date"], inplace=True)

    # Reset the index
    df_filled.reset_index(drop=True, inplace=True)
    df_filled.sort_values(by="Date", inplace=True)

    # Assuming df is your dataframe and 'Date' is in the correct datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Create a copy of the dataframe to perform the operations

    # Replace 0 values with NaN for backward fill
    df_filled.replace(0, pd.NA, inplace=True)

    # Specify columns that need to be backward filled
    integer_columns = ["Minimum", "Maximum", "Average"]

    # Group by 'Commodity' and apply backward fill within each group
    df_filled[integer_columns] = df_filled.groupby("Commodity")[integer_columns].bfill()

    return df_filled
