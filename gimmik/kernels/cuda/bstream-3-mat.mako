<%inherit file='base'/>

__global__ void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const int ldb = ${ldb // width};
    const int ldc = ${ldc // width};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < n)
    {
        ${dtype} bv;

## Iterare through the used rows of B
% for kx in bix:
        bv = __ldcg(b + i + ${kx}*ldb);
  % for j, jx in enumerate(A[:, kx]):
    % if jx != 0 and kx == afix[j]:
        ${dtype} csub_${j} = ${jx}*bv;
    % elif jx != 0:
        csub_${j} += ${jx}*bv;
    % endif
  % endfor
% endfor

## Handle rows of A which are all zero
% for j, jx in enumerate(afix):
  % if jx == -1 and beta == 0:
        ${dtype} csub_${j} = make_zero();
  % endif
% endfor

## Iterare through the used rows of B
% for kx in bix:
  % for j, jx in enumerate(A_1[:, kx]):
    % if jx != 0 and kx == afix_1[j]:
        ${dtype} csub_${j}_1 = ${jx}*csub_${kx};
    % elif jx != 0:
        csub_${j}_1 += ${jx}*csub_${kx};
    % endif
  % endfor
% endfor

## Iterare through the used rows of B
% for kx in bix:
  % for j, jx in enumerate(A_2[:, kx]):
    % if jx != 0 and kx == afix_2[j]:
        ${dtype} csub_${j}_2 = ${jx}*csub_${kx}_1;
    % elif jx != 0:
        csub_${j}_2 += ${jx}*csub_${kx}_1;
    % endif
    ##
    % if kx == alix[j] and beta == 0:
        __stcg(c + i + ${j}*ldc, csub_${j}_2);
    % elif kx == alix[j] and beta == 1:
        c[i + ${j}*ldc] += csub_${j}_2;
    % elif kx == alix[j]:
        c[i + ${j}*ldc] = csub_${j}_2 + ${beta}*c[i + ${j}*ldc];
    % endif
  % endfor
% endfor

## Handle rows of A which are all zero
% for j, jx in enumerate(afix):
  % if jx == -1 and beta == 0:
        c[i + ${j}*ldc] = make_zero();
  % elif jx == -1 and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
  % endif
% endfor
    }
}
