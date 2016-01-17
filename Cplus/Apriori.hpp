#ifndef Experiments_Apriori_hpp
#define Experiments_Apriori_hpp

#include "Library.hpp"

class Apriori
{
protected:
    unsigned    n_item;
    
public:
    Apriori(const unsigned n_item)
    :n_item(n_item)
    {
        ;
    }
    
    set<set<unsigned>> run(const vector<set<unsigned>>& dataset, unsigned support)
    {
        set<set<unsigned>>      result;
        
        vector<unsigned>        counts_item(n_item);
        for(auto item=0; item<dataset.size(); ++item)
        {
            for(auto subitem=dataset[item].begin(); subitem!=dataset[item].end(); ++subitem)
            {
                ++ counts_item[*subitem];
            }
        }
        for(auto item=0; item<n_item; ++item)
        {
            if (counts_item[item] >= support)
            {
                set<unsigned>   set_item;
                set_item.insert(item);
                result.insert(set_item);
            }
        }
        
        for(auto level=0; level<n_item; ++level)
        {
            set<set<unsigned>>   current_result;
            for(auto item=result.begin(); item!=result.end(); ++item)
            {
                for(unsigned add_item=0; add_item<n_item; ++add_item)
                {
                    if (item->find(add_item) == item->end())
                    {
                        set<unsigned>   current_extending = *item;
                        current_extending.insert(add_item);
                        
                        unsigned current_support = 0;
                        for(auto dataitem=dataset.begin(); dataitem!=dataset.end(); ++dataitem)
                        {
                            bool check_contained = true;
                            for(auto ce=current_extending.begin(); ce!=current_extending.end(); ++ce)
                            {
                                if (dataitem->count(*ce) == 0)
                                {
                                    check_contained = false;
                                    break;
                                }
                            }
                            
                            if (check_contained)
                            {
                                ++ current_support;
                            }
                        }
                        
                        if (current_support >= support)
                        {
                            current_result.insert(current_extending);
                        }
                    }
                }
                
                if (current_result.size() == 0)
                {
                    break;
                }
                
                result.insert(current_result.begin(), current_result.end());
            }
        }
        
        return result;
    }
};

#endif
