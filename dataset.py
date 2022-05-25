#!/usr/bin/env python3
import numpy as np

def generate_halfmoon(n1, n2, max_angle=np.pi):
    alpha = np.linspace(0, max_angle, n1)
    beta = np.linspace(0, max_angle, n2)
    X1 = np.vstack([np.cos(alpha), np.sin(alpha)]) + 0.1 * np.random.randn(2,n1)
    X2 = np.vstack([1 - np.cos(beta), 1 - np.sin(beta) - 0.5]) + 0.1 * np.random.randn(2,n2)
    y1, y2 = -np.ones(n1), np.ones(n2)
    return X1, y1, X2, y2


if __name__ == "__main__":
    X1, y1, X2, y2 = generate_halfmoon(n1=100, n2=100, max_angle=2)
    np.savez('halfmoon.npz', X1=X1, y1=y1, X2=X2, y2=y2)
    