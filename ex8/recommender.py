from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt

def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat['Y'], mat['R']


def loadParams(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat['X'], mat['Theta'], mat['num_users'], mat['num_movies'], mat['num_features']


def loadMoviesID(path):
    movie_list = []

    with open(path, encoding='latin-1') as f:
        for line in f:
            tokens = line.strip().split(' ')
            movie_list.append(' '.join(tokens[1:]))

    movie_list = np.array(movie_list)
    return movie_list


def serialize(X, thetas):
    """serialize 2 matrix"""
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), thetas.ravel()))


def deserialize(params, num_movies=1682, num_users=943, num_features=10):
    """into ndarray of X(1682, 10), theta(943, 10)"""
    return params[:num_movies * num_features].reshape(num_movies, num_features), \
           params[num_movies * num_features:].reshape(num_users, num_features)


def costFunc(params, Y, R, num_movies=1682, num_users=943, num_features=10):
    X, thetas = deserialize(params, num_movies, num_users, num_features)
    tmp = np.multiply(X @ thetas.T - Y, R)
    tmp = np.power(tmp, 2)
    return np.sum(tmp) / 2


def regularizedCostFunc(params, Y, R, lamda=1, num_movies=1682, num_users=943, num_features=10):
    cost = costFunc(params, Y, R, num_movies, num_users, num_features)
    regularization = np.sum(params**2) / 2 * lamda
    return cost +regularization


def gradient(params, Y, R, num_movies=1682, num_users=943, num_features=10):
    X, thetas = deserialize(params, num_movies, num_users, num_features)
    inner = np.multiply(R, (X @ thetas.T - Y))
    gradient_X = inner @ thetas
    gradient_thetas = inner.T @ X
    return serialize(gradient_X, gradient_thetas)


def regularizedGradient(params, Y, R, lamda=1, num_movies=1682, num_users=943, num_features=10):
    gradients = gradient(params, Y, R, num_movies, num_users, num_features)
    gradients += lamda * params
    return gradients


def main():
    dataPath = 'data/ex8_movies.mat'
    Y, R = loadData(dataPath)
    print(Y.shape, R.shape)
    paramsPath = 'data/ex8_movieParams.mat'
    X, thetas, num_users, num_movies, num_featues = loadParams(paramsPath)
    print(X.shape, thetas.shape, num_users, num_movies, num_featues)
    params = serialize(X, thetas)

    # use some examples to test the function
    users = 4
    movies = 5
    features = 3

    X_sub = X[:movies, :features]
    theta_sub = thetas[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]

    param_sub = serialize(X_sub, theta_sub)
    print(param_sub.shape)
    print(costFunc(param_sub, Y_sub, R_sub, num_movies=movies, num_users=users, num_features=features))
    print(costFunc(params, Y, R))

    print(regularizedCostFunc(param_sub, Y_sub, R_sub, lamda=1.5, num_movies=movies, num_users=users, num_features=features))
    print(regularizedCostFunc(params, Y, R))

    # training the recommender system based on your preference, or you can change the RATINGS array to affact
    # the result of recommender system.
    # show the movies id
    moviesPath = 'data/movie_ids.txt'
    movie_list = loadMoviesID(moviesPath)
    print(loadMoviesID(moviesPath)[:5])

    # this represents your preference.
    ratings = np.zeros(1682)

    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5

    # prepare the data.
    Y = np.insert(Y, 0, ratings, axis=1)
    R = np.insert(R, 0, ratings != 0, axis=1)
    n_features = 50
    n_movie, n_user = Y.shape
    l = 10
    X = np.random.standard_normal((n_movie, n_features))
    theta = np.random.standard_normal((n_user, n_features))
    param = serialize(X, theta)

    # normalized ratings
    Y_norm = Y - Y.mean()
    # train the model and it's slow.
    res = opt.minimize(fun=regularizedCostFunc,
                       x0=param,
                       args=(Y_norm, R, l, n_movie, n_user, n_features),
                       method='TNC',
                       jac=regularizedGradient)
    print(res)

    X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
    # predict the ratings for the user[0]
    prediction = X_trained @ theta_trained.T
    my_preds = prediction[:, 0] + Y.mean()
    print(my_preds.shape)
    # Descending order
    idx = np.argsort(my_preds)[::-1]
    # select the top ten to recommend.
    print(len(movie_list))
    for id in idx[:10]:
        print(movie_list[id])


if __name__ == '__main__':
    main()

