#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct{
  float *speeds0;
  float *speedsN;
  float *speedsS;
  float *speedsW;
  float *speedsE;
  float *speedsNW;
  float *speedsNE;
  float *speedsSW;
  float *speedsSE;
} t_speed_arr;

kernel void accelerate_flow(const global int* obstacles,
                            const int nx, const int ny,
                            const float density, const float accel,
                            global float* speedsW,
                            global float* speedsNW,
                            global float* speedsSW,
                            global float* speedsE,
                            global float* speedsNE,
                            global float* speedsSE)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;
  /* get column index */
  int ii = get_global_id(0);
  int index = ii + jj*nx;
  // printf("%f \n",speedsW[index]);

  /* if the cell is not occupied and
  ** we don't send a negative density */

  float a = speedsW[index] - w1;
  float b = speedsNW[index] - w2;
  float c = speedsSW[index] - w2;
  w1 = (!obstacles[index] && a >0.f && b>0.f && c>0.f) ? w1 : 0;
  w2 = (!obstacles[index]  && a >0.f && b>0.f && c>0.f) ?w2 : 0;


/* increase 'east-side' densities */
speedsE[index] += w1;
speedsNE[index] += w2;
speedsSE[index] += w2;
/* decrease 'west-side' densities */
speedsW[index] -= w1;
speedsNW[index] -= w2;
speedsSW[index] -= w2;


}

kernel void collision( global float* speeds0,
                      global float* speedsN,
                      global float* speedsS,
                      global float* speedsW,
                      global float* speedsE,
                      global float* speedsNW,
                      global float* speedsNE,
                      global float* speedsSW,
                      global float* speedsSE,
                      const global int* obstacles,
                      const int nx, const float omega,
                      local float* local_cell_sums,
                      local float* local_totu_sums,
                      global int* global_cell_sums,
                      global float* global_totu_sums,
                      const int blksize,const int ny)
{
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tot_cells = 0;
    float tot_u = 0.0f;
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    int index = ii + jj*nx;

    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    float speeds[9];

    /*
    ** 6 2 5
    **  \|/
    ** 3-0-1
    **  /|\
    ** 7 4 8
    **
    */

    //copy into private memory
    speeds[0] = speeds0[ii+jj*nx];
    speeds[1] = speedsE[x_w + jj*nx];
    speeds[2] = speedsN[ii + y_s*nx];
    speeds[3] = speedsW[x_e + jj*nx];
    speeds[4] = speedsS[ii + y_n*nx];
    speeds[5] = speedsNE[x_w + y_s*nx];
    speeds[6] = speedsNW[x_e + y_s*nx];
    speeds[7] = speedsSW[x_e + y_n*nx];
    speeds[8] = speedsSE[x_w + y_n*nx];

    /* compute local density total */
    float local_density = 0.f;

    local_density += speeds[0];
    local_density += speeds[2];
    local_density += speeds[4];
    local_density += speeds[3];
    local_density += speeds[1];
    local_density += speeds[6];
    local_density += speeds[5];
    local_density += speeds[7];
    local_density += speeds[8];
    /* compute x velocity component */
    float u_x = (speeds[1]
                  + speeds[5]
                  + speeds[8]
                  - (speeds[3]
                     + speeds[6]
                     + speeds[7])  )
                 / local_density;
    /* compute y velocity component */
    float u_y = (speeds[2]
                  + speeds[5]
                  + speeds[6]
                  - (speeds[4]
                     + speeds[7]
                     + speeds[8]) )
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

    /* don't consider occupied cells */
   speeds0[index] = (obstacles[index]) ? speeds[0] : speeds[0] + omega * (d_equ[0] - speeds[0]);
   speedsE[index] = (obstacles[index]) ? speeds[3] : speeds[1] + omega * (d_equ[1] - speeds[1]);
   speedsN[index] = (obstacles[index]) ? speeds[4] : speeds[2] + omega * (d_equ[2] - speeds[2]);
   speedsW[index] = (obstacles[index]) ? speeds[1] : speeds[3] + omega * (d_equ[3] - speeds[3]);
   speedsS[index] = (obstacles[index]) ? speeds[2] : speeds[4] + omega * (d_equ[4] - speeds[4]);
   speedsNE[index] = (obstacles[index]) ? speeds[7] : speeds[5] + omega * (d_equ[5] - speeds[5]);
   speedsNW[index] = (obstacles[index]) ? speeds[8] : speeds[6] + omega * (d_equ[6] - speeds[6]);
   speedsSW[index] = (obstacles[index]) ? speeds[5] : speeds[7] + omega * (d_equ[7] - speeds[7]);
   speedsSE[index] = (obstacles[index]) ? speeds[6] : speeds[8] + omega * (d_equ[8] - speeds[8]);
    tot_u = (!obstacles[index]) ? sqrt(u_sq) : 0;
    tot_cells = (!obstacles[index]);

    int local_id = lx + ly * blksize;
    local_cell_sums[local_id] = tot_cells;
    local_totu_sums[local_id] = tot_u;

    int gx = get_group_id(0);
    int gy = get_group_id(1);
    int group_id = gx + gy * (nx/blksize);

    int cell_sum = 0;
    float totu_sum;
    int num_wrk_items = get_local_size(0) * get_local_size(1);
    for(int offset =  1; offset < num_wrk_items; offset*= 2)
    {
        int mask = 2*offset -1;
        barrier(CLK_LOCAL_MEM_FENCE);
        int x = (local_id&mask);
        int a = (x==0) ?local_cell_sums[local_id+ offset] : 0 ;
        float b = (x==0)?local_totu_sums[local_id+ offset] : 0.f;
        local_cell_sums[local_id] += a;
        local_totu_sums[local_id] += b;
    }
    if(local_id == 0)
    {
        global_cell_sums[group_id] = local_cell_sums[0];
        global_totu_sums[group_id] = local_totu_sums[0];
    }
}

kernel void av_velocity(global float* speeds0,
                      global float* speedsN,
                      global float* speedsS,
                      global float* speedsW,
                      global float* speedsE,
            global float* speedsNW,
           global float* speedsNE,
            global float* speedsSW,
            global float* speedsSE,
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

       local_density += speeds0[index];
       local_density += speedsN[index];
       local_density += speedsS[index];
       local_density += speedsW[index];
       local_density += speedsE[index];
       local_density += speedsNW[index];
       local_density += speedsNE[index];
       local_density += speedsSW[index];
       local_density += speedsSE[index];
       /* compute x velocity component */
       float u_x = (speedsE[index]
                     + speedsNE[index]
                     + speedsSE[index]
                     - (speedsW[index]
                        + speedsNW[index]
                        + speedsSW[index])  )
                    / local_density;
       /* compute y velocity component */
       float u_y = (speedsN[index]
                     + speedsNE[index]
                     + speedsNW[index]
                     - (speedsS[index]
                        + speedsSW[index]
                        + speedsSE[index]) )
                    / local_density;
       /* accumulate the norm of x- and y- velocity components */
       tot_u = sqrt((u_x * u_x) + (u_y * u_y));
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
