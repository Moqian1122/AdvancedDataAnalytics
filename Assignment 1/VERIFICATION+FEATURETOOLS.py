#FEATURE TOOLS
if extra_params['feature_tools'] == 1:
    combined_df = pd.concat([X_train, X_val, X_test], ignore_index=True)
    es = ft.EntitySet(id='EntitySet')

    # Add your DataFrame as an entity to the EntitySet
    es = es.add_dataframe(
        dataframe_name="AllDataFT",
        dataframe=combined_df,
        index = "index"
    )

    # Perform Deep Feature Synthesis (DFS)
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='AllDataFT',
                                      trans_primitives=['add_numeric', 'multiply_numeric'],
                                      max_depth=1)

    # Split the combined dataset back into train, validation, and test sets
    train_size = len(X_train)
    val_size = len(X_val)

    X_train = feature_matrix[:train_size]
    X_val = feature_matrix[train_size:train_size + val_size]
    X_test = feature_matrix[train_size + val_size:]
    X_train['target'] = y_train

    #Drop variables (there are too many >2000)
    #Start by dropping uncorrelated variables to target
    correlation_matrix_init = X_train[X_train.select_dtypes(include=['number']).columns.tolist()].corr()
    corrtarget_init = correlation_matrix_init['target'].reset_index()
    drop_init = []
    for i in range(0, len(corrtarget_init)):
        if abs(corrtarget_init['target'][i]) <= 0.05:
            drop_init.append(corrtarget_init['index'][i])
    X_train = X_train.drop(columns = drop_init)
    X_val = X_val.drop(columns = drop_init)
    X_test = X_test.drop(columns = drop_init)

    #Drop clusters of highly correlated variables
    correlation_matrix_ft = X_train[X_train.select_dtypes(include=['number']).columns.tolist()].corr()
    corrtarget_ft = correlation_matrix_ft['target']

    high_correlation_pairs_ft = []
    for i in range(len(correlation_matrix_ft.columns)):
        for j in range(i+1, len(correlation_matrix_ft.columns)):
            if abs(correlation_matrix_ft.iloc[i, j]) > extra_params["corr_thresh2"]:
                high_correlation_pairs_ft.append((correlation_matrix_ft.columns[i], correlation_matrix_ft.columns[j]))
    connected_groups_ft = find_connected_components(high_correlation_pairs_ft)
    variables_to_drop_ft = [get_variables_to_drop(group, corrtarget_ft) for group in connected_groups_ft]
    drop_vars_ft = [item for sublist in variables_to_drop_ft for item in sublist]

    X_train = X_train.drop(columns = 'target')
    X_train = X_train.drop(columns = drop_vars_ft)
    X_val = X_val.drop(columns = drop_vars_ft)
    X_test = X_test.drop(columns = drop_vars_ft)

#VERIFICATION 

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Initialize KFold cross-validator with the same random seed
kf = KFold(n_splits=3, shuffle = True, random_state=random_seed)

# Manually split the data into folds using KFold
manual_splits = list(kf.split(X_train))

rf = RandomForestClassifier()
for i, (train_split, val_split) in enumerate(manual_splits):
    xtrain_set = X_train.iloc[train_split, :]
    ytrain_set = y_train[train_split]
    xval_set = X_train.iloc[val_split, :]
    yval_set = y_train[val_split]

    rf.fit(xtrain_set, ytrain_set)

    y_pred = rf.predict(xval_set)
    y_pred_proba = rf.predict_proba(xval_set)
    y_pred_proba = [sublist[1] for sublist in y_pred_proba]
    
    df_results = pd.DataFrame()
    df_results['y_pred'] = y_pred
    df_results['y_true'] = yval_set
    df_results['predict_proba'] = y_pred_proba
    df_results['average cost min'] = list(averagecostmintrain[val_split])

    sorted_df = df_results.sort_values(by= "predict_proba", ascending = False)
    
    profittopk = 0
    for i in range(0, 20):
        if sorted_df['y_true'][i] == sorted_df['y_pred'][i]:
            
            profittopk += sorted_df['average cost min'].iloc[i]
    fpr, tpr, thresholds = metrics.roc_curve(yval_set, y_pred)
    AUC = metrics.auc(fpr, tpr)
    model_score = 0.7*profittopk + 0.3*AUC
    print(model_score) #THIS SHOULD RETURN SCORES THAT ARE IDENTICAL TO THE ONES RETURNED IN THE SKLEARN KFOLD CROSS VALIDATION