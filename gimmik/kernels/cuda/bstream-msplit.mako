<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
%>

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
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    ${dtype} bv, csub[${-(-m // msplit)}];
    __shared__ ${dtype} bsub[2][${bsz}][${blockx}];

    if (i >= n)
      return;

## Iterate over each row-chunk of C
% for cid, mcx in enumerate(mx):
    if (threadIdx.y == ${cid})
    {
  ## Iterate over each row-chunk of B
  % for bb in range(len(bchunks)):
    ## Fill the initial shared memory block
    % if loop.first:
      % for kx in bchunks[0]:
        % if loop.index % msplit == cid:
        bsub[0][${loop.index}][threadIdx.x] = __ldcg(b + i + ${kx}*ldb);
        % endif
      % endfor
        __barrier_sync(0);
    % endif
    ## Start filling the next shared memory block
    % if not loop.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
        bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x] = __ldcg(b + i + ${kx}*ldb);
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][threadIdx.x];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
        csub[${j}] = ${jx}*bv;
        % elif jx != 0:
        csub[${j}] += ${jx}*bv;
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]] and beta == 0:
        __stcg(c + i + ${mcx[j]}*ldc, csub[${j}]);
        % elif kx == alix[mcx[j]] and beta == 1:
        c[i + ${mcx[j]}*ldc] += csub[${j}];
        % elif kx == alix[mcx[j]]:
        c[i + ${mcx[j]}*ldc] = csub[${j}] + ${beta}*c[i + ${mcx[j]}*ldc];
        % endif
      % endfor
    % endfor
        __barrier_sync(0);
  % endfor
  ## Handle rows of A which are all zero
  % for j, jx in enumerate(afix):
    % if jx == -1 and j % msplit == cid and beta == 0:
        __stcg(c + i + ${j}*ldc, make_zero());
    % elif jx == -1 and j % msplit == cid and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
    % endif
  % endfor
    }
% endfor
}
