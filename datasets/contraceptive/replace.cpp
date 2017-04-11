#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>

using namespace std;

#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define all(v) (v).begin(), (v).end()
#define uniq(v) (v).erase(unique(all(v)), v.end())
#define IOS ios::sync_with_stdio(0);

#define fr(a, b, c) for(int (a) = (b); (a) < (c); ++(a))
#define rp(a, b) fr(a,0,b)
#define cl(a, b) memset((a), (b), sizeof(a))
#define sc(a) scanf("%d", &a)
#define sc2(a, b) scanf("%d %d", &a, &b)
#define sc3(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define pr(a) printf("%d\n", a);
#define pr2(a, b) printf("%d %d\n", a, b)
#define pr3(a, b, c) printf("%d %d %d\n", a, b, c)
#define IOS ios::sync_with_stdio(0);
#define sz(v) (int) (v).size()

typedef unsigned long long ull;
typedef long long ll;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef pair<ll, ll> pll;

vector<unordered_map<string, string> > dict(10);

int main()
{
	//Wife's Education
	dict[1].insert(mp("1", "low"));
	dict[1].insert(mp("2", "avg-low"));
	dict[1].insert(mp("3", "avg-high"));
	dict[1].insert(mp("4", "high"));

	//Husband Education
	dict[2].insert(mp("1", "low"));
	dict[2].insert(mp("2", "avg-low"));
	dict[2].insert(mp("3", "avg-high"));
	dict[2].insert(mp("4", "high"));

	//Number of children ever born
	dict[3].insert(mp("zero", "0"));
	dict[3].insert(mp("one", "1"));
	dict[3].insert(mp("two", "2"));
	dict[3].insert(mp("three", "3"));
	dict[3].insert(mp("four", "4"));
	dict[3].insert(mp("five", "5"));
	dict[3].insert(mp("six", "6"));
	dict[3].insert(mp("seven", "7"));
	dict[3].insert(mp("eight", "8"));
	dict[3].insert(mp("nine", "9"));
	dict[3].insert(mp("ten", "10"));
	dict[3].insert(mp("eleven", "11"));
	dict[3].insert(mp("twelve", "12"));
	dict[3].insert(mp("thirteen", "13"));
	dict[3].insert(mp("fourteen", "14"));

	//Wife's religion
	dict[4].insert(mp("0", "Non-Islam"));
	dict[4].insert(mp("1", "Islam"));

	//Wife's now working?
	dict[5].insert(mp("0", "Yes"));
	dict[5].insert(mp("1", "No"));

	//Husband's occupation type
	dict[6].insert(mp("1", "one"));
	dict[6].insert(mp("2", "two"));
	dict[6].insert(mp("3", "three"));
	dict[6].insert(mp("4", "four"));

	//Standard-of-living index
	dict[7].insert(mp("1", "low"));
	dict[7].insert(mp("2", "avg-low"));
	dict[7].insert(mp("3", "avg-high"));
	dict[7].insert(mp("4", "high"));

	//Media Exposure
	dict[8].insert(mp("0", "Good"));
	dict[8].insert(mp("1", "Not Good"));

	//Contraceptive method used
	dict[9].insert(mp("0", "No-use"));
	dict[9].insert(mp("1", "Long-term"));
	dict[9].insert(mp("2", "Short-term"));

	ifstream in; in.open("wrong.csv");
	ofstream out; out.open("contraceptive.csv");
	string line;
	while(getline(in, line))
	{
		vector<string> fields;
		boost::split(fields, line, boost::is_any_of(","));
		out << fields[0];
		fr(i, 1, sz(fields))
		{
			out << ",";
			if(i == 3)
				out << dict[i][fields[i]];
			else
				out << fields[i];
		}
		out << endl;
	}
	return 0;
}
