export function parseAndRound(number, decimalPlaces){
    return parseFloat(parseFloat( number ).toFixed( decimalPlaces ))
}