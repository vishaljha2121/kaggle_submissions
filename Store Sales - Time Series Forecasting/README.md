# TIME SERIES FORECAST USING HYBRID MACHINE LEARNING MODEL

Creating a python notebook for a Kaggle competition wherein I built a model to accurately predict the unit sales for thousands of items sold at different stores of a chain.

A training dataset of dates, stores, and item information, promotion and unit sales was provided.

The evaluation for the submission file for this competition is done using **ROOT MEAN SQUARED LOGARITHMIC ERROR**

	⎷1nn∑i=1(log(1+^yi)−log(1+yi))2

## Requirements

 - pandas
 - matplotlib
 - numpy
 - sklearn
 - pickle

## Data cleaning
A feature of distinction between old stores and new stores (old stores are ones that had sales in January 2013) was created to be included in the model.  While processing dates of sales, multiindexing was used to get christmas days into time index to avoid issue for algorithms which require multiindexing for ever date, family and store. A calendar was created to include special holidays wherein sales took place.

## Model
I used a BoostedHybrid model because it allows using two models which can use a time-based feature to extrapolate long-term and seasonal trends and also simultaneously use features to find more complex interrelationships.

For model 1: LinearRegression and model 2 : XGBRegression
because of their ability to extract seasonal trends and complex interrelationships including lags respectively.

I split the fit function for the models into 2 methods, to produce y_residual from fit1 and create a lag feature which can be used to further train the model to identify complex interrelations.

    class BoostedHybridModel:
	    def __init__(self, model_1, model_2):
	        self.model_1 = model_1
	        self.model_2 = model_2
	        self.y_columns = None
	        self.stack_cols = None
	        self.y_resid = None

	    def fit1(self, X_1, y, stack_cols=None):
	        self.model_1.fit(X_1, y) 
	        y_fit = pd.DataFrame(
	            self.model_1.predict(X_1), 
	            index=X_1.index,
	            columns=y.columns,
	        )
	        self.y_resid = y - y_fit 
	        self.y_resid = self.y_resid.stack(stack_cols).squeeze()  
	        
	    def fit2(self, X_2, first_n_rows_to_ignore, stack_cols=None):
	        self.model_2.fit(X_2.iloc[first_n_rows_to_ignore*1782: , :], self.y_resid.iloc[first_n_rows_to_ignore*1782:]) 
	        self.y_columns = y.columns 
	        self.stack_cols = stack_cols 

	    def predict(self, X_1, X_2, first_n_rows_to_ignore):
	        y_pred = pd.DataFrame(
	            self.model_1.predict(X_1.iloc[first_n_rows_to_ignore: , :]),
	            index=X_1.iloc[first_n_rows_to_ignore: , :].index,
	            columns=self.y_columns,
	        )
	        y_pred = y_pred.stack(self.stack_cols).squeeze()  
	        y_pred += self.model_2.predict(X_2.iloc[first_n_rows_to_ignore*1782: , :]) 
	        return y_pred.unstack(self.stack_cols)

To deal with the NaNs created from lag features, a max_lag functions is used which passes first n rows to ignore so that the appropriate number of rows are ignored in fit2 and predict methods. By doing this, the rows can be reused over and over for methods like rolling mean.

For feature generation for the two fit methods, two seperate feature generation functions are used X1 and X2.
For time-series features, deterministic process is used where we can add more Fourier terms of different types.

    def make_det_proc_features(df):
	    y = df.loc[:, 'sales']
	    fourier_m = CalendarFourier(freq='M', order=4)
	    dp = DeterministicProcess(
	        index=y.index,
	        constant=True,
	        order=1,
	        seasonal=True,
	        additional_terms=[fourier_m],
	        drop=True,
	    )
	    return y, dp

For model 2, a lot of features are required for to capture more complex relationships.

    def encode_categoricals(df, columns):
	    le = LabelEncoder()
	    for col in columns:
	        df[col] = le.fit_transform(df[col])
	    return df
    
    def make_X2_lags(ts, lags, lead_time=1, name='y', stack_cols=None):
	    ts = ts.unstack(stack_cols)
	    df = pd.concat(
	        {
	            f'{name}_lag_{i}': ts.shift(i, freq="D") for i in range(lead_time, lags + lead_time)
	        },
	        axis=1
	    )

	    df = df.stack(stack_cols).reset_index()
	    df = encode_categoricals(df, stack_cols)
	    df = df.set_index('date').sort_values(by=stack_cols)

	    return df

## Validation

The forecasting method I used in this notebook is day by day refit all days methods where,

> The model creates the forecast one day at a time so that the next day's lag can be calculated and used for following days. However, with each new forecast, the entire set of days (training days AND forecast days) are refitted. For a 16 day forecast, day one will therefore be forecast 16 different times, though the forecasts are only very slightly different with each iteration through the loop.

    X_1_train, y_train, dp_val = make_X1_features(store_sales_in_date_range, train_start_day, train_end_day)
    model_for_val.fit1(X_1_train, y_train, stack_cols=['store_nbr', 'family'])

    X_2_train = make_X2_features(store_sales_in_date_range.drop('sales', axis=1).stack(['store_nbr', 'family']), model_for_val.y_resid)
    model_for_val.fit2(X_2_train, max_lag, stack_cols=['store_nbr', 'family'])

    y_fit = model_for_val.predict(X_1_train, X_2_train, max_lag).clip(0,0)
    dp_for_full_X1_val_date_range = dp_val.out_of_sample(steps=validation_days)

    for step in range(validation_days):
        dp_steps_so_far = dp_for_full_X1_val_date_range.loc[val_start_day:val_start_day+pd.Timedelta(days=step),:]

        X_1_combined_dp_data = pd.concat([dp_val.in_sample(), dp_steps_so_far])
        X_2_combined_data = pd.concat([store_sales_in_date_range,
                                       store_data_in_val_range.loc[val_start_day:val_start_day+pd.Timedelta(days=step), :]])
        
        X_1_val = make_X1_features(X_1_combined_dp_data, train_start_day, val_start_day+pd.Timedelta(days=step), is_test_set=True)
        X_2_val = make_X2_features(X_2_combined_data.drop('sales', axis=1).stack(['store_nbr', 'family']), model_for_val.y_resid)

        y_pred_combined = model_for_val.predict(X_1_val, X_2_val, max_lag).clip(0.0)

        y_plus_y_val = pd.concat([y_train, y_pred_combined.iloc[-(step+1):]])
        model_for_val.fit1(X_1_val, y_plus_y_val, stack_cols=['store_nbr', 'family'])
        model_for_val.fit2(X_2_val, max_lag, stack_cols=['store_nbr', 'family'])

        rmsle_valid = mean_squared_log_error(y_val.iloc[step:step+1], y_pred_combined.iloc[-1:]) ** 0.5
        print(f'Validation RMSLE: {rmsle_valid:.5f}', "for", val_start_day+pd.Timedelta(days=step))

    y_pred = y_pred_combined[val_start_day:val_end_day]
    print("\ny_pred: ")
    display(y_pred.apply(lambda s: truncateFloat(s)))

    if type(model_for_val.model_2) == XGBRegressor:
        pickle.dump(model_for_val.model_2, open("xgb_temp.pkl", "wb"))
        m2 = pickle.load(open("xgb_temp.pkl", "rb"))

        print("XGBRegressor paramaters:\n",m2.get_xgb_params(), "\n")


## Result visualtisation
For intuitive and effective understanding of the forecast, graph plots were used to visualise the results

The plots in notebook lets us see the what the validation and test look against the original data.

## RESULT
My submission gave a score of 0.51141 on Kaggle witha leaderboard position of 658 (as of 10-12-2021).

