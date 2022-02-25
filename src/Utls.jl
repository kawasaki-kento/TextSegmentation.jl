module Utls
    using Languages
    stop_words = stopwords(Languages.English())

    mutable struct SentenceElements
        elements_dct::Dict{String,Int}
    end
    
    function tokenize(sentence)
        sentence = replace(sentence, r"[“”\"\"();:,.'?!]" => "")
        sentence = lowercase(sentence)
        return split.(sentence, " ")
    end

    function count_elements(sequence)
        elements_dct = Dict{String,Int}()
        for i in sequence
            if !in(i, stop_words)
                if i in keys(elements_dct)
                    elements_dct[i] += 1
                else
                    elements_dct[i] = 1
                end
            end
        end
        return elements_dct
    end

    function merge_elements(se::SentenceElements, dct)
        for i in keys(dct)
            if i in keys(se.elements_dct)
                se.elements_dct[i] += dct[i]
            else
                se.elements_dct[i] = dct[i]
            end
        end
    end

    function calculate_cosin_similarity(elements_dct_1, elements_dct_2)
        n1 = sqrt(sum([i * i for i in values(elements_dct_1)]))
        n2 = sqrt(sum([i * i for i in values(elements_dct_2)]))
        common_key = intersect(keys(elements_dct_1), keys(elements_dct_2))

        if isempty(common_key)
            num = 0
        else
            num = sum(elements_dct_1[k] * elements_dct_2[k] for k in common_key)
        end

        try
            if n1 * n2 < 1e-9
                return 0
            end
            return num / (n1 * n2)
        catch
            return 0
        end
    end

end