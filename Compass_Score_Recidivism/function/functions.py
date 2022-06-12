{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1fddd0f2-bc04-4d6f-9f0f-ab7d9910d9a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def runGridSearch(X_train, y_train):\n",
    "    from sklearn.model_selection import GridSearchCV\n",
    "    # Number of trees in random forest\n",
    "    # Using num=2 will significantly reduce run time (~3 min vs ~65 min)\n",
    "    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 2)]\n",
    "    # Using num=10 will significantly increase run time\n",
    "    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]\n",
    "    # Number of features at every split\n",
    "    max_features = ['auto', 'sqrt']\n",
    "    # Maximum number of levels in tree\n",
    "    # Using num=2 will significantly reduce run time (~3 min vs ~65 min)\n",
    "    max_depth = [int(x) for x in np.linspace(10, 110, num = 2)]\n",
    "    # Using num=10 will significantly increase run time\n",
    "    #max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]\n",
    "    max_depth.append(None)\n",
    "    # Minimum number of samples required to split a node\n",
    "    min_samples_split = [2, 5, 10]\n",
    "    # Minimum number of samples required at each leaf node\n",
    "    min_samples_leaf = [1, 2, 4]\n",
    "    # Method of selecting samples for training each tree\n",
    "    bootstrap = [True, False]\n",
    "    # Create the random grid\n",
    "    grid = {'n_estimators': n_estimators,\n",
    "                   'max_features': max_features,\n",
    "                   'max_depth': max_depth,\n",
    "                   'min_samples_split': min_samples_split,\n",
    "                   'min_samples_leaf': min_samples_leaf,\n",
    "                   'bootstrap': bootstrap}\n",
    "\n",
    "    # Create a regressor using values from grid\n",
    "    rf_reg = RandomForestClassifier()\n",
    "    # Instantiate the grid search model\n",
    "    grid_search = GridSearchCV(estimator = rf_reg, param_grid = grid, cv = 3, n_jobs = -1, verbose = 1)\n",
    "\n",
    "    # Train the classifier\n",
    "    best_grid = grid_search.fit(X_train, y_train)\n",
    "\n",
    "    # Make predictions\n",
    "    grid_preds = best_grid.predict(X_test)\n",
    "\n",
    "    print(\"The best hyperparameters found during the grid search are:\")\n",
    "    print(best_grid.best_params_)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
