import numpy as np
class ErsemMixin:
    @property
    def attrs4d(self):
        '''
        Update attributes_dict from ERSEM xarray.Dataset (self.ds)
        
        Returns
        -------
        attributes_dict : dict
            Updated attributes_dict
        '''
        
        attributes_dict = {
            var: {'units': self.ds[var].attrs.get('units', None), 
                  'long_name': self.ds[var].attrs.get('long_name', None)}
            for var in self.ds.data_vars
            if self.ds[var].dims == ('time','z','lat','lon') or self.ds[var].dims == ('time','z','y','x')
        }

        # Edit units.
        # Replace 'm^3' with 'm$^3$'. 
        for var, attrs in attributes_dict.items():
            units = attrs.get('units', '')
            if 'm^3' in units:
                attributes_dict[var]['units'] = units.replace('m^3', 'm$^3$')

        # Replace 'm-3' with 'm$^{-3}$'.
        for var, attrs in attributes_dict.items():
            units = attrs.get('units', '')
            if 'm-3' in units:
                attributes_dict[var]['units'] = units.replace('m-3', 'm$^{-3}$')

        # Replace '1/m' with 'm$^{-1}$'.
        for var, attrs in attributes_dict.items():
            units = attrs.get('units', '')
            if '1/m' in units:
                attributes_dict[var]['units'] = units.replace('1/m', 'm$^{-1}$')

        # Replace 'm-1' with 'm$^{-1}$'.
        for var, attrs in attributes_dict.items():
            units = attrs.get('units', '')
            if 'm-1' in units:
                attributes_dict[var]['units'] = units.replace('m-1', 'm$^{-1}$')
        
        # Edit long_name.
        for var, attrs in attributes_dict.items():
            long_name = attrs.get('long_name', '')
            if 'potential temperature' in long_name:
                attributes_dict[var]['long_name'] = long_name.replace('potential temperature', 'Temperature')

        # Add vmin and vmax as 'range' in attributes_dict, rounded to sensible values
        for var in attributes_dict:
            data = self.ds[var].values
            vmin, vmax = np.min(data), np.max(data)
            
            # キリのよい値に調整（整数部分に丸める）
            vmin_rounded = np.floor(vmin)
            vmax_rounded = np.ceil(vmax)

            # vmin と vmax の差が約20程度になるように調整
            #range_span = vmax_rounded - vmin_rounded
            #if range_span < 20:
            #    padding = (20 - range_span) / 2
            #    vmin_rounded -= padding
            #    vmax_rounded += padding
            
            # 'range' キーとして (vmin_rounded, vmax_rounded) のタプルを追加
            attributes_dict[var]['range'] = (float(vmin_rounded), float(vmax_rounded))

        return attributes_dict
