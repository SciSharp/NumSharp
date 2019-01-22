using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.Sparse
{
    public partial class csr_matrix
    {
        public void sort_indices()
        {
            int indptrstart = 0;
            int indptrend = 0;
            int tmpindicesvalue = 0;
            double tmpdatavalue = 0;

            for (int i = 1; i < indptr.size; i++)
            {
                indptrend = indptr.Data<int>()[i] - 1;

                for (int j = indptrstart; j < indptrend; j++)
                {
                    
                    if (indices.Data<int>()[j + 1] < indices.Data<int>()[j])
                    {
                        // switch indices
                        tmpindicesvalue = indices.Data<int>()[j];
                        indices.Data<int>()[j] = indices.Data<int>()[j + 1];
                        indices.Data<int>()[j + 1] = tmpindicesvalue;

                        // switch value
                        tmpdatavalue = data.Data<double>()[j];
                        data.Data<double>()[j] = data.Data<double>()[j + 1];
                        data.Data<double>()[j + 1] = tmpdatavalue;
                    }
                }

                indptrstart = indptr.Data<int>()[i];
            }

            has_sorted_indices = true;
        }
    }
}
