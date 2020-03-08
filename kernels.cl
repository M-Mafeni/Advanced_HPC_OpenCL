#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed_arr* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);
  int index = ii + jj*params.nx;

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[index]
      && (cells->speedsW[index] - w1) > 0.f
      && (cells->speedsNW[index] - w2) > 0.f
      && (cells->speedsSW[index] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells->speedsE[index] += w1;
    cells->speedsNE[index] += w2;
    cells->speedsSE[index] += w2;
    /* decrease 'west-side' densities */
    cells->speedsW[index] -= w1;
    cells->speedsNW[index] -= w2;
    cells->speedsSW[index] -= w2;
  }
}

kernel void propagate(global t_speed_arr* cells,
                      global t_speed_arr* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  int index= ii + jj*params.nx;
  tmp_cells->speeds0[index] = cells->speeds0[ii+jj*params.nx]; /* central cell, no movement */
  tmp_cells->speedsE[index] = cells->speedsE[x_w + jj*params.nx]; /* east */
  tmp_cells->speedsN[index] = cells->speedsN[ii + y_s*params.nx]; /* north */
  tmp_cells->speedsW[index] = cells->speedsW[x_e + jj*params.nx]; /* west */
  tmp_cells->speedsS[index] = cells->speedsS[ii + y_n*params.nx]; /* south */
  tmp_cells->speedsNE[index] = cells->speedsNE[x_w + y_s*params.nx]; /* north-east */
  tmp_cells->speedsNW[index] = cells->speedsNW[x_e + y_s*params.nx]; /* north-west */
  tmp_cells->speedsSW[index] = cells->speedsSW[x_e + y_n*params.nx]; /* south-west */
  tmp_cells->speedsSE[index] = cells->speedsSE[x_w + y_n*params.nx]; /* south-east */
}
kernel void rebound(global t_speed_arr* cells,
                      global t_speed_arr* tmp_cells,
                      global int* obstacles,
                      int nx)
{
    /* get column and row indices */
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    int index = ii + jj*params.nx;
    if (obstacles[jj*params.nx + ii])
     {
       /* called after propagate, so taking values from scratch space
       ** mirroring, and writing into main grid */
       cells->speedsE[index] = tmp_cells->speedsW[index];
       cells->speedsN[index] = tmp_cells->speedsS[index];
       cells->speedsW[index] = tmp_cells->speedsE[index];
       cells->speedsS[index] = tmp_cells->speedsN[index];
       cells->speedsNE[index] = tmp_cells->speedsSW[index];
       cells->speedsNW[index] = tmp_cells->speedsSE[index];
       cells->speedsSW[index] = tmp_cells->speedsNE[index];
       cells->speedsSE[index] = tmp_cells->speedsNW[index];
     }
}

kernel void collision(global t_speed_arr* cells,
                      global t_speed_arr* tmp_cells,
                      global int* obstacles,
                      int nx, float omega
                      )
{
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */

    /* don't consider occupied cells */
    if (!obstacles[ii + jj*nx])
    {
      /* compute local density total */
      float local_density = 0.f;

      int index = ii + jj*params.nx;
      local_density += tmp_cells->speeds0[index];
      local_density += tmp_cells->speedsN[index];
      local_density += tmp_cells->speedsS[index];
      local_density += tmp_cells->speedsW[index];
      local_density += tmp_cells->speedsE[index];
      local_density += tmp_cells->speedsNW[index];
      local_density += tmp_cells->speedsNE[index];
      local_density += tmp_cells->speedsSW[index];
      local_density += tmp_cells->speedsSE[index];
      /* compute x velocity component */
      float u_x = (tmp_cells->speedsE[index]
                    + tmp_cells->speedsNE[index]
                    + tmp_cells->speedsSE[index]
                    - (tmp_cells->speedsW[index]
                       + tmp_cells->speedsNW[index]
                       + tmp_cells->speedsSW[index])  )
                   / local_density;
      /* compute y velocity component */
      float u_y = (tmp_cells->speedsN[index]
                    + tmp_cells->speedsNE[index]
                    + tmp_cells->speedsNW[index]
                    - (tmp_cells->speedsS[index]
                       + tmp_cells->speedsSW[index]
                       + tmp_cells->speedsSE[index]) )
                   / local_density;

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                 * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                       + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                       + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                       + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                       + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                       + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                       + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                       + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                       + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                       - u_sq / (2.f * c_sq));

      /* relaxation step */
      // for (int kk = 0; kk < NSPEEDS; kk++)
      // {
      //   cells[ii + jj*nx].speeds[kk] = tmp_cells[ii + jj*nx].speeds[kk]
      //                                           + omega
      //                                           * (d_equ[kk] - tmp_cells[ii + jj*nx].speeds[kk]);
      // }

      cells->speeds0[index] = tmp_cells->speeds0[index] + params.omega * (d_equ[0] - tmp_cells->speeds0[index]);
        cells->speedsE[index] = tmp_cells->speedsE[index] + params.omega * (d_equ[1] - tmp_cells->speedsE[index]);
        cells->speedsN[index] = tmp_cells->speedsN[index] + params.omega * (d_equ[2] - tmp_cells->speedsN[index]);
        cells->speedsW[index] = tmp_cells->speedsW[index] + params.omega * (d_equ[3] - tmp_cells->speedsW[index]);
        cells->speedsS[index] = tmp_cells->speedsS[index] + params.omega * (d_equ[4] - tmp_cells->speedsS[index]);
        cells->speedsNE[index] = tmp_cells->speedsNE[index] + params.omega * (d_equ[5] - tmp_cells->speedsNE[index]);
        cells->speedsNW[index] = tmp_cells->speedsNW[index] + params.omega * (d_equ[6] - tmp_cells->speedsNW[index]);
        cells->speedsSW[index] = tmp_cells->speedsSW[index] + params.omega * (d_equ[7] - tmp_cells->speedsSW[index]);
        cells->speedsSE[index] = tmp_cells->speedsSE[index] + params.omega * (d_equ[8] - tmp_cells->speedsSE[index]);
    }
}

kernel void av_velocity(global t_speed_arr* cells,
    global int* obstacles,
    int nx,
    local float* local_cell_sums,
    local float* local_totu_sums,
    global int* global_cell_sums,
    global float* global_totu_sums,
    const int blksize
    )
{
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tot_cells = 0;
    float tot_u = 0.0f;

    int nblocks = nx/blksize;

    int index = ii+jj*nx;
    if (!obstacles[index])
    {
        /* local density total */
       float local_density = 0.f;

       local_density += cells->speeds0[index];
       local_density += cells->speedsN[index];
       local_density += cells->speedsS[index];
       local_density += cells->speedsW[index];
       local_density += cells->speedsE[index];
       local_density += cells->speedsNW[index];
       local_density += cells->speedsNE[index];
       local_density += cells->speedsSW[index];
       local_density += cells->speedsSE[index];
       /* compute x velocity component */
       float u_x = (cells->speedsE[index]
                     + cells->speedsNE[index]
                     + cells->speedsSE[index]
                     - (cells->speedsW[index]
                        + cells->speedsNW[index]
                        + cells->speedsSW[index])  )
                    / local_density;
       /* compute y velocity component */
       float u_y = (cells->speedsN[index]
                     + cells->speedsNE[index]
                     + cells->speedsNW[index]
                     - (cells->speedsS[index]
                        + cells->speedsSW[index]
                        + cells->speedsSE[index]) )
                    / local_density;
       /* accumulate the norm of x- and y- velocity components */
       tot_u = sqrtf((u_x * u_x) + (u_y * u_y));
       /* increase counter of inspected cells */
       ++tot_cells;
    }
    int local_id = lx + ly * blksize;
    local_cell_sums[local_id] = tot_cells;
    local_totu_sums[local_id] = tot_u;

    barrier(CLK_LOCAL_MEM_FENCE);
    int gx = get_group_id(0);
    int gy = get_group_id(1);
    int group_id = gx + gy * (nx/blksize);

    int cell_sum = 0;
    float totu_sum;
    int num_wrk_items = get_local_size(0) * get_local_size(1);
    if(local_id == 0){
        totu_sum = 0;
        for(int i = 0; i < num_wrk_items; i++){
            cell_sum += local_cell_sums[i];
            totu_sum += local_totu_sums[i];
        }
        global_cell_sums[group_id] = cell_sum;
        global_totu_sums[group_id] = totu_sum;
    }
}
