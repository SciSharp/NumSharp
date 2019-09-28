namespace NumSharp.UnitTest.Others
{
    public class LinearRegression {
        private float alpha;
        private int n_iter;
        private int n_samples;
        private int n_features;
        private NDArray X;
        private NDArray y;
        private NDArray @params;
        private NDArray coef_;
        private float? intercept_;

        public LinearRegression(NDArray X, NDArray y, float alpha = 0.03f, int n_iter = 1500) {
            this.alpha = alpha;
            this.n_iter = n_iter;
            this.n_samples = y.size;
            this.n_features = np.size(X, 1);
            this.X = np.hstack(np.ones(this.n_samples, 1), 
                                    (X - np.mean(X, 0) / np.std(X, 0)));
            this.y = np.expand_dims(y, -1);
            this.@params = np.zeros((this.n_features + 1, 1), NPTypeCode.Single);
            this.coef_ = null;
            this.intercept_ = null;
        }

        public LinearRegression fit() {
            for (int i = 0; i < this.n_iter; i++) {
                this.@params = this.@params - (this.alpha / this.n_samples) *
                               np.matmul(this.X.T, np.matmul(this.X, this.@params) - this.y);
            }
            
            this.intercept_ = @params.GetSingle(0);
            this.coef_ = @params["1:"];
            
            return this;
        }

        public float score(NDArray X = null, NDArray y = null) {
            if (X is null)
                X = this.X;
            else {
                n_samples = np.size(X, 0);
                this.X = np.hstack(np.ones(this.n_samples, 1), 
                    (X - np.mean(X, 0) / np.std(X, 0)));
            }

            if (y is null) {
                y = this.y;
            } else
                y = np.expand_dims(y, -1);

            var y_pred = np.matmul(X, @params);
            var score = 1 - ((np.power((y - y_pred), 2)).sum() / (np.power(y - y.mean(), 2)).sum());

            return score;
        }

        public NDArray predict(NDArray X) {
            n_samples = np.size(X, 0);
            y = np.matmul(
                    np.hstack(np.ones(this.n_samples, 1), (X - np.mean(X, 0) / np.std(X, 0))), 
                    @params
                );

            return y;
        }

        public NDArray get_params() 
            => @params;
    }
}
