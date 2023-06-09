SECRET_KEY = 'rs6zn8ps#se2-*5a&xl@(=qj&ky5g411_t8o27f_0=+us=*nv5'
DEBUG = True

INSTALLED_APPS = (
    'django_microproject',
    'corsheaders',
)

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
]

ROOT_URLCONF = 'django_microproject.urls'

ALLOWED_HOSTS = ['*']
CORS_ORIGIN_ALLOW_ALL = True
