try: 
    from scipy import percentile
except ImportError:
    # if the scipy version is too old...
    from external_libs.scipy_compat.percentile import percentile
