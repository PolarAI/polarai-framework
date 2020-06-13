#pragma once

#include <string>

static std::string textKernels = R"(
int get_global_id_linear() {
  int workDim = get_work_dim();
  if (workDim == 1) {
    return get_global_id(0);
  } else if (workDim == 2) {
    return get_global_id(1) + get_global_size(1) * get_global_id(0);
  } else if (workDim == 3) {
    return get_global_id(2) + get_global_size(2) * get_global_id(1) +
           get_global_size(2) * get_global_size(1) * get_global_id(0);
  }

  return 0;
}


kernel void ffill(float pattern, const ulong size, global float* out) {
  int id = get_global_id_linear();
//  printf("FFILL: Id: %d ; Size: %lu; ", id, size);
  if (id < size) {
    out[get_global_id_linear()] = pattern;
//    printf("Pattern: %f;\n", pattern);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fadd(global const float* a, float scaleA, global const float* b,
                 float scaleB, const ulong size, global float* out) {
  const int id = get_global_id_linear();
//  printf("FADD: Id: %d; Size: %lu; A: %p; B: %p; Out: %p; ", id, size, a, b, out);
  if (id < size) {
    out[id] = a[id] * scaleA + b[id] * scaleB;
//    printf("%f * %f + %f * %f = %f;\n", scaleA, a[id], scaleB, b[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fcopy(global const float* inp, const ulong size, global float* out) {
  const int id = get_global_id_linear();
//  printf("FCOPY: Id: %d; Size: %lu; ", id, size);
  if (id < size) {
    out[id] = inp[id];
//    printf("Value: %f;\n", out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fdivide(global const float* numerator, global const float* denominator, const ulong size, global float* out) {
  const int id = get_global_id_linear();
//  printf("FDIVIDE: Id: %d; Size: %lu; Numerator: %p; Denominator: %p; Out: %p;  ", id, size, numerator, denominator, out);
  if (id < size) {
    out[id] = numerator[id] / denominator[id];
//    printf("%.10f / %.10f = %.10f;\n", numerator[id], denominator[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void flogloss(global const float* predicted, global const float* groundTruth, const ulong size, global float* out) {
  const int id = get_global_id_linear();
//  printf("FLOGLOSS: Id: %d; Size: %lu; Predicted: %p; GroundTruth: %p; Out: %p; ", id, size, predicted, groundTruth, out);
  float eps = 1e-5;
  if (id < size) {
    out[id] = -groundTruth[id] * log(predicted[id] + eps) -
              (1 - groundTruth[id]) * log(1 - predicted[id] + eps);
//    printf("predicted: %f; groundTruth: %f; out: %f;\n", predicted[id],
//           groundTruth[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fmatmul(const ulong transLeft, const ulong transRight, const ulong m, const ulong n, const ulong k,
                    global const float* left, global const float* right, global float* out) {
  const int id = get_global_id_linear();
//  printf("FMATMUL: Id: %d; TrLeft: %lu; TrRight: %lu; M: %lu; N: %lu; K: %lu; LeftPtr: %p; RightPtr: %p; ", id, transLeft, transRight, m, n, k, left, right);
  if (id < m * n) {
    ulong indexRow = id / n;
    ulong indexColumn = id % n;
//    printf("\nIdRow: %lu; IdCol: %lu; ", indexRow, indexColumn);
    ulong leftIncrement = 0;
    ulong leftInd = 0;
    if (transLeft == 0) { // Not transposed
      leftIncrement = 1;
      leftInd = indexRow * k;
    } else { // Transposed
      leftIncrement = m;
      leftInd = indexRow;
    }
    ulong rightIncrement = 0;
    ulong rightInd = 0;
    if (transRight == 0) { // Not transposed
      rightIncrement = n;
      rightInd = indexColumn;
    } else { // Transposed
      rightIncrement = 1;
      rightInd = indexColumn * k;
    }
//    printf("LeftInc: %lu; RightInc: %lu; ", leftIncrement, rightIncrement);
//    printf("LeftIndex: %lu; RightIndex: %lu; ", leftInd, rightInd);
    out[indexRow * n + indexColumn] = 0;
    for (int iteration = 0; iteration < k; ++iteration, leftInd += leftIncrement, rightInd += rightIncrement) {
      out[indexRow * n + indexColumn] += left[leftInd] * right[rightInd];
//      printf("\nOut: %f; Left: %f; Right: %f", out[indexRow * n + indexColumn], left[leftInd], right[rightInd]);
    }
  } else {
//    printf("NOOP");
  }
//  printf("\n");
}

kernel void fmul(global const float* a, global const float* b, const ulong size, global float* out) {
  const int id = get_global_id_linear();
//  printf("FMUL: Id: %d; Size: %lu; A: %p; B: %p; Out: %p; ", id, size, a, b, out);
  if (id < size) {
    out[id] = a[id] * b[id];
//    printf("%f * %f = %f;\n", a[id], b[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fmulconcat(global const float* gradient, const ulong gradientSize, global const float* localDerivative, const ulong localDerivativeSize, global float* out) {
  const int id = get_global_id_linear();
//  printf("FMULCONCAT: Id: %d; GradientSize: %lu; LocalDerivativeSize: %lu; GradPtr: %p; LocDerivPtr: %p; ", id, gradientSize, localDerivativeSize, gradient, localDerivative);
  if (id < localDerivativeSize) {
    out[id] = gradient[0] * localDerivative[id];
//    printf("(grad) %f * (locderiv)%f = %f;\n", gradient[0], localDerivative[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}

kernel void fsigmoid(global const float* input, const ulong size, global float* out) {
  const int id = get_global_id_linear();
  float eps = 1e-5;
//  printf("FSIGMOID: Id: %d; Size: %lu; Input: %p; Output: %p; ", id, size, input, out);
  if (id < size) {
    out[id] = 1 / (1 + exp(-input[id]));
    if (fabs(out[id]-1) < eps) {
      out[id] = 1 - eps;
    } else if (fabs(out[id]) < eps) {
      out[id] = eps;
    }
//    printf("sigm(%f) = %f;\n", input[id], out[id]);
  } else {
//    printf("NOOP\n");
  }
}
)";
