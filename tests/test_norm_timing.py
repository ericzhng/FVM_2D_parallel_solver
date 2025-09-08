import numpy as np
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result

    return wrapper


face_normal = np.array([0.5, 0.5, 0.5])


@timing_decorator
def np_norm():
    for k in range(1000):
        normal = np.linalg.norm(face_normal)
    return normal


@timing_decorator
def my_norm():
    for k in range(1000):
        normal2 = np.sqrt(
            face_normal[0] ** 2 + face_normal[1] ** 2 + face_normal[2] ** 2
        )
    return normal2


print(np_norm())
print(my_norm())
